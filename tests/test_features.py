import json
import threading

import httpx
import openai
import pytest

from inference_lib.client import (
    INDEX_KEY,
    SAMPLE_KEY,
    BatchStats,
    SamplingParams,
    Validation,
    format_summary,
    merge_jsonl,
    parse_json_output,
    row_messages,
    run_batch,
    summarize,
)
from inference_lib.multinode import _strip_nodes, plan_assignment
from inference_lib.stamp import check_resume, fingerprint, write_stamp
from inference_lib.watchdog import ServerPool


class Fake:
    """Scriptable stand-in for openai.OpenAI.  ``script(call_no, messages, kw)`` returns
    (content, finish_reason) or raises."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.lock = threading.Lock()
        self.chat = self
        self.completions = self

    def create(self, model, messages, **kw):
        with self.lock:
            self.calls.append((messages, kw))
            n = len(self.calls)
        content, finish = self.script(n, messages, kw)
        msg = type("M", (), {"content": content, "reasoning": None})
        choice = type("C", (), {"message": msg, "finish_reason": finish})
        usage = type("U", (), {"prompt_tokens": 3, "completion_tokens": 5})
        return type("R", (), {"choices": [choice], "usage": usage})


def healthy_pool(n=1, **kw):
    return ServerPool([f"http://fake{i}/v1" for i in range(n)], healthy=lambda url: True, log=lambda m: None, **kw)


def run(fake, rows, **kw):
    kw.setdefault("pool", healthy_pool())
    return run_batch(
        client=fake,
        rows=rows,
        model="m",
        params=kw.pop("params", SamplingParams()),
        use_tqdm=False,
        log=lambda m: None,
        **kw,
    )


# ---------------------------------------------------------------- validation + corrective retry


def test_parse_json_output_tolerates_fences_and_prose():
    assert parse_json_output('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_output('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(json.JSONDecodeError):
        parse_json_output("no json here")


def test_invalid_json_is_reasked_with_correction():
    def script(n, messages, kw):
        return ("{broken", "stop") if n == 1 else ('{"score": 2}', "stop")

    fake = Fake(script)
    [out] = run(fake, [{"prompt": "p"}], validation=Validation(retry_on=frozenset({"invalid-json"}), json_output=True))
    assert out["parsed"] == {"score": 2} and out["error"] is None
    assert out["attempts"] == 2 and out["retry_reasons"] == ["invalid-json"]
    second = fake.calls[1][0]
    assert second[-2] == {"role": "assistant", "content": "{broken"} and "rejected" in second[-1]["content"]


def test_schema_violation_fails_after_retries_and_is_resumable(tmp_path):
    schema = {"type": "object", "required": ["score"], "properties": {"score": {"type": "integer", "maximum": 4}}}
    fake = Fake(lambda n, m, kw: ('{"score": 9}', "stop"))
    out_path = tmp_path / "o.jsonl"
    v = Validation(retry_on=frozenset({"schema"}), json_output=True, schema=schema, max_retries=2)
    [out] = run(fake, [{"prompt": "p"}], validation=v, jsonl_path=out_path)
    assert out["error"].startswith("schema:") and out["attempts"] == 3 and len(fake.calls) == 3
    fake2 = Fake(lambda n, m, kw: ('{"score": 3}', "stop"))
    [out2] = run(fake2, [{"prompt": "p"}], validation=v, jsonl_path=out_path)
    assert out2["error"] is None and out2["parsed"] == {"score": 3}


def test_length_retry_doubles_max_tokens_up_to_cap():
    fake = Fake(lambda n, m, kw: ("x", "length") if kw["max_tokens"] < 400 else ("done", "stop"))
    [out] = run(
        fake,
        [{"prompt": "p"}],
        params=SamplingParams(max_tokens=100),
        validation=Validation(retry_on=frozenset({"length"}), max_retries=3, max_tokens_cap=400),
    )
    assert [c[1]["max_tokens"] for c in fake.calls] == [100, 200, 400]
    assert out["responses"] == "done" and out["retry_reasons"] == ["length", "length"]


def test_length_without_retry_is_kept_as_success():
    fake = Fake(lambda n, m, kw: ("partial", "length"))
    [out] = run(fake, [{"prompt": "p"}])
    assert out["error"] is None and out["finish_reason"] == "length" and out["attempts"] == 1


def test_custom_validator():
    fake = Fake(lambda n, m, kw: ("no" if n == 1 else "yes", "stop"))
    v = Validation(
        retry_on=frozenset({"validator"}), custom=lambda row, res: None if res["response"] == "yes" else "say yes"
    )
    [out] = run(fake, [{"prompt": "p"}], validation=v)
    assert out["responses"] == "yes" and out["retry_reasons"] == ["validator"]


def test_empty_is_reasked_by_default_reason():
    fake = Fake(lambda n, m, kw: ("" if n == 1 else "ok", "stop"))
    [out] = run(fake, [{"prompt": "p"}], validation=Validation(retry_on=frozenset({"empty"})))
    assert out["responses"] == "ok" and out["attempts"] == 2


# ---------------------------------------------------------------- n samples + multi-turn


def test_n_samples_with_seed_offsets_and_resume(tmp_path):
    fake = Fake(lambda n, m, kw: (f"s{kw['seed']}", "stop"))
    out_path = tmp_path / "o.jsonl"
    rows = run(
        fake, [{"prompt": "a"}, {"prompt": "b"}], n_samples=3, params=SamplingParams(seed=10), jsonl_path=out_path
    )
    assert [(r[INDEX_KEY], r[SAMPLE_KEY]) for r in rows] == [(i, s) for i in range(2) for s in range(3)]
    assert {r["responses"] for r in rows if r[INDEX_KEY] == 0} == {"s10", "s11", "s12"}
    fake2 = Fake(lambda n, m, kw: ("again", "stop"))
    run(fake2, [{"prompt": "a"}, {"prompt": "b"}], n_samples=3, jsonl_path=out_path)
    assert fake2.calls == []  # nothing left to do


def test_multi_turn_messages_rows():
    row = {
        "messages": json.dumps(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
                {"role": "user", "content": "again"},
            ]
        )
    }
    assert [m["role"] for m in row_messages(row, "sys")] == ["system", "user", "assistant", "user"]
    fake = Fake(lambda n, m, kw: (str(len(m)), "stop"))
    [out] = run(fake, [row])
    assert out["responses"] == "3"


# ---------------------------------------------------------------- watchdog


def _conn_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "http://fake/v1/chat/completions"))


def test_outage_is_recovered_without_spending_row_attempts():
    state = {"down": True, "restarts": 0}

    def script(n, m, kw):
        if state["down"]:
            raise _conn_error()
        return "ok", "stop"

    def recover(i):
        state["restarts"] += 1
        state["down"] = False

    pool = ServerPool(["http://fake/v1"], recover=recover, healthy=lambda u: not state["down"], log=lambda m: None)
    rows = run(Fake(script), [{"prompt": str(i)} for i in range(5)], pool=pool, retries=1, concurrency=5)
    assert all(r["error"] is None for r in rows) and state["restarts"] == 1
    assert all(r["attempts"] == 1 for r in rows)


def test_request_error_on_healthy_server_counts_attempts():
    fake = Fake(lambda n, m, kw: (_ for _ in ()).throw(_conn_error()) if n < 3 else ("ok", "stop"))
    [out] = run(fake, [{"prompt": "p"}], retries=3)
    assert out["error"] is None and out["attempts"] == 3


def test_unrecoverable_server_aborts_resumably(tmp_path):
    pool = ServerPool(
        ["http://fake/v1"], recover=lambda i: None, max_restarts=1, healthy=lambda u: False, log=lambda m: None
    )
    fake = Fake(lambda n, m, kw: (_ for _ in ()).throw(_conn_error()))
    stats = BatchStats()
    rows = run(fake, [{"prompt": str(i)} for i in range(4)], pool=pool, jsonl_path=tmp_path / "o.jsonl", stats=stats)
    assert stats.aborted and "restarts" in stats.aborted and rows == []


# ---------------------------------------------------------------- stamp / summary / multi-node


def _fp(version="0.30.0", image="sha256:aaa", temp=0.0):
    servers = [
        {
            "vllm_version": version,
            "models": [{"id": "m", "root": "/w", "max_model_len": 1}],
            "container": {"image_id": image},
        }
    ]
    return fingerprint(servers, {"preset": "glm", "preset_sha256": "x"}, {"temperature": temp}, {}, 1, None)


def test_resume_refuses_a_different_stack(tmp_path):
    out = tmp_path / "o.jsonl"
    write_stamp(out, {"fingerprint": _fp()})
    assert check_resume(out, _fp(), allow_mixed=False) is not None
    with pytest.raises(SystemExit, match="vllm_versions"):
        check_resume(out, _fp(version="0.1.dev20051"), allow_mixed=False)
    with pytest.raises(SystemExit, match="sampling"):
        check_resume(out, _fp(temp=0.7), allow_mixed=False)
    stored = check_resume(out, _fp(image="sha256:bbb"), allow_mixed=True)
    assert stored["mixed_with"][0]["diff"]["image_ids"]["now"] == ["sha256:bbb"]


def test_summary_counts():
    rows = [
        {"responses": "a", "finish_reason": "stop", "completion_tokens": 10, "prompt_tokens": 2, "retry_reasons": []},
        {
            "responses": "b",
            "finish_reason": "length",
            "completion_tokens": 30,
            "prompt_tokens": 2,
            "retry_reasons": ["length"],
        },
        {"responses": None, "error": "schema: bad", "completion_tokens": 0, "prompt_tokens": 2},
    ]
    s = summarize(rows, stats=BatchStats(submitted=3, wall_seconds=4.0, completion_tokens=40))
    assert (s["ok"], s["failed"], s["truncated"], s["re_asked"]) == (2, 1, 1, 1)
    assert s["errors_by_type"] == {"schema": 1} and s["this_run"]["completion_tok_per_s"] == 10.0
    assert "truncated (finish=length) 1" in format_summary(s)
    multi = {**s, "this_run": {"nodes": ["a", "b"], "wall_seconds": 5.0, "worker_exit_codes": {"a": 0, "b": 0}}}
    assert "2 nodes" in format_summary(multi)
    trunc_empty = summarize([{"responses": "", "finish_reason": "length"}, {"responses": "", "finish_reason": "stop"}])
    assert (trunc_empty["truncated_no_answer"], trunc_empty["empty"]) == (1, 1)


def test_plan_assignment_covers_todo_exactly_once():
    a = plan_assignment(n_rows=5, n_samples=2, done={(0, 0), (3, 1)}, nodes=["n1", "n2", "n3"])
    got = [tuple(k) for v in a.values() for k in v]
    assert sorted(got) == sorted((i, s) for i in range(5) for s in range(2) if (i, s) not in {(0, 0), (3, 1)})
    assert len(got) == len(set(got))


def test_merge_prefers_success_across_shards(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({INDEX_KEY: 0, "responses": None, "error": "x"}) + "\n")
    b.write_text(
        json.dumps({INDEX_KEY: 0, "responses": "ok"}) + "\n" + json.dumps({INDEX_KEY: 1, "responses": "k"}) + "\n"
    )
    rows = merge_jsonl([a, b], tmp_path / "out.jsonl")
    assert [r["responses"] for r in rows] == ["ok", "k"]


def test_strip_nodes():
    assert _strip_nodes(["batch", "--nodes", "a,b", "-m", "x", "--nodes=c"]) == ["batch", "-m", "x"]
