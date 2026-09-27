import json

from inference_lib.client import INDEX_KEY, SamplingParams, compact_jsonl, completed_indices, run_batch


class _Msg:
    def __init__(self, content):
        self.content = content
        self.reasoning = "r"


class _Fake:
    """Minimal stand-in for openai.OpenAI; fails prompts listed in ``fail``."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, model, messages, **kw):
        prompt = messages[-1]["content"]
        self.calls.append((prompt, kw))
        if prompt in self.fail:
            raise RuntimeError("boom")
        choice = type("C", (), {"message": _Msg(prompt.upper()), "finish_reason": "stop"})
        usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 2})
        return type("R", (), {"choices": [choice], "usage": usage})


def test_sampling_params_body():
    kw = SamplingParams(max_tokens=5, temperature=0, top_k=20, enable_thinking=False, seed=1).request_kwargs()
    assert kw["max_tokens"] == 5 and kw["seed"] == 1
    assert kw["extra_body"] == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}


def test_batch_resume_retries_failed_rows_only(tmp_path):
    rows = [{"prompt": f"p{i}"} for i in range(6)]
    out = tmp_path / "o.jsonl"
    fake = _Fake(fail={"p2"})
    res = run_batch(
        client=fake,
        rows=rows,
        model="m",
        params=SamplingParams(),
        retries=1,
        jsonl_path=out,
        use_tqdm=False,
        log=lambda m: None,
    )
    assert [r[INDEX_KEY] for r in res] == list(range(6))
    assert res[2]["error"] and res[3]["responses"] == "P3" and res[3]["reasoning"] == "r"
    assert completed_indices(out) == {0, 1, 3, 4, 5}

    fake2 = _Fake()
    res2 = run_batch(
        client=fake2,
        rows=rows,
        model="m",
        params=SamplingParams(),
        retries=1,
        jsonl_path=out,
        use_tqdm=False,
        log=lambda m: None,
    )
    assert [c[0] for c in fake2.calls] == ["p2"]  # only the failed row is redone
    assert [r["responses"] for r in res2] == [f"P{i}" for i in range(6)]
    assert len(out.read_text().splitlines()) == 6


def test_compact_tolerates_torn_line(tmp_path):
    out = tmp_path / "o.jsonl"
    out.write_text(
        json.dumps({INDEX_KEY: 1, "responses": "b"})
        + "\n"
        + json.dumps({INDEX_KEY: 0, "responses": "a"})
        + '\n{"_idx": 2, "resp'
    )
    compact_jsonl(out)
    assert [json.loads(line)[INDEX_KEY] for line in out.read_text().splitlines()] == [0, 1]
