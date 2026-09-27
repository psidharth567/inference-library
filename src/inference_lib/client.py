"""OpenAI-compatible client helpers and the concurrent, validated, resumable batch runner."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import openai
from openai import OpenAI
from tqdm import tqdm

from .watchdog import ServerDead, ServerPool

PROMPT_KEYS = ("prompts", "prompt", "text", "input", "question", "query", "instruction", "content")
SYSTEM_KEYS = ("system", "system_prompt", "system_message", "sys")
MESSAGES_KEY = "messages"
INDEX_KEY = "_idx"
SAMPLE_KEY = "_sample"
RETRY_REASONS = ("empty", "length", "invalid-json", "schema", "validator")


# ------------------------------------------------------------------ rows -> messages


def get_prompt(row: dict[str, Any]) -> str:
    for k in PROMPT_KEYS:
        if row.get(k) is not None and str(row[k]).strip():
            return str(row[k])
    raise KeyError(f"each row needs one of {PROMPT_KEYS} or '{MESSAGES_KEY}'; got keys {list(row)[:6]}")


def get_system(row: dict[str, Any], global_system: str | None = None) -> str | None:
    for k in SYSTEM_KEYS:
        if row.get(k) is not None and str(row[k]).strip():
            return str(row[k]).strip()
    return global_system.strip() if global_system and global_system.strip() else None


def build_messages(*, system: str | None, prompt: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return messages


def get_row_messages(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """A row's ``messages`` (list, or a JSON string as in CSV cells), else None."""
    msgs = row.get(MESSAGES_KEY)
    if msgs is None or msgs == "":
        return None
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    if not isinstance(msgs, list) or not all(isinstance(m, dict) and "role" in m for m in msgs):
        raise ValueError("'messages' must be a list of {role, content} objects")
    return msgs


def row_messages(row: dict[str, Any], global_system: str | None = None) -> list[dict[str, Any]]:
    """Chat messages for a row: its multi-turn ``messages`` (a system prompt from the row or
    the global one is prepended when the list has none), else system + prompt."""
    msgs = get_row_messages(row)
    system = get_system(row, global_system)
    if msgs is None:
        return build_messages(system=system, prompt=get_prompt(row))
    if system and not any(m.get("role") == "system" for m in msgs):
        return [{"role": "system", "content": system}, *msgs]
    return list(msgs)


# ------------------------------------------------------------------ client + params


def get_client(base_url: str, api_key: str | None = None, timeout: float = 3600.0) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=timeout,
        max_retries=0,
        http_client=httpx.Client(
            limits=httpx.Limits(max_connections=4096, max_keepalive_connections=4096), timeout=timeout
        ),
    )


def resolve_model_id(client: OpenAI, model: str | None = None) -> str:
    """The served model id: ``model`` if the server serves it, else the server's only model."""
    ids = [m.id for m in client.models.list().data]
    if not ids:
        raise RuntimeError("server reports no models")
    if model and model in ids:
        return model
    if model:
        for i in ids:
            if i.split("/")[-1].lower() == model.split("/")[-1].lower():
                return i
    return ids[0]


@dataclass
class SamplingParams:
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    enable_thinking: bool | None = None  # chat_template_kwargs for Qwen3/Qwen3.5/GLM/DSV4-style templates
    response_format: dict[str, Any] | None = None  # {"type": "json_object"} or a json_schema format
    extra_body: dict[str, Any] = field(default_factory=dict)

    def request_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"max_tokens": self.max_tokens, "temperature": self.temperature}
        if self.top_p is not None:
            kw["top_p"] = self.top_p
        if self.seed is not None:
            kw["seed"] = self.seed
        if self.response_format is not None:
            kw["response_format"] = self.response_format
        extra = dict(self.extra_body)
        if self.top_k is not None:
            extra["top_k"] = self.top_k
        if self.enable_thinking is not None:
            extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = self.enable_thinking
        if extra:
            kw["extra_body"] = extra
        return kw

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, {}, [])}


def chat(client: OpenAI, *, model: str, messages: list[dict[str, Any]], params: SamplingParams) -> dict[str, Any]:
    """One chat completion -> {response, reasoning, finish_reason, prompt_tokens, completion_tokens}."""
    c = client.chat.completions.create(model=model, messages=messages, **params.request_kwargs())
    choice = c.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    content = msg.content or ""
    if reasoning:  # reasoning parsers leave the newlines that followed </think>
        content = content.lstrip()
    return {
        "response": content,
        "reasoning": reasoning,
        "finish_reason": choice.finish_reason,
        "prompt_tokens": c.usage.prompt_tokens if c.usage else None,
        "completion_tokens": c.usage.completion_tokens if c.usage else None,
    }


def generate_one(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    return chat(
        client,
        model=model,
        messages=build_messages(system=system, prompt=prompt),
        params=SamplingParams(max_tokens=max_tokens, temperature=temperature),
    )["response"]


# ------------------------------------------------------------------ output validation


def parse_json_output(text: str) -> Any:
    """Parse a JSON answer, tolerating ``` fences and text around one JSON value."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.rsplit("```", 1)[0]
    try:
        return json.loads(s)
    except json.JSONDecodeError as first:
        for i, ch in enumerate(s):
            if ch in "{[":
                try:
                    return json.JSONDecoder().raw_decode(s[i:])[0]
                except json.JSONDecodeError:
                    continue
        raise first


def json_schema_format(schema: dict[str, Any], name: str = "output") -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


@dataclass
class Validation:
    """Which outputs count as failures and get re-asked (up to ``max_retries`` extra attempts).

    retry_on: subset of RETRY_REASONS.  ``invalid-json``/``schema`` need ``json_output``;
    ``validator`` needs ``custom`` (row, result) -> error message or None.
    On ``length`` the next attempt doubles max_tokens (up to ``max_tokens_cap``).
    With ``corrective`` a JSON/schema/validator failure is re-asked with the rejected answer and
    the reason appended to the conversation."""

    retry_on: frozenset[str] = frozenset()
    max_retries: int = 2
    json_output: bool = False
    schema: dict[str, Any] | None = None
    custom: Callable[[dict[str, Any], dict[str, Any]], str | None] | None = None
    corrective: bool = True
    max_tokens_cap: int | None = None

    def __post_init__(self) -> None:
        unknown = set(self.retry_on) - set(RETRY_REASONS)
        if unknown:
            raise ValueError(f"unknown retry reasons {sorted(unknown)}; choose from {RETRY_REASONS}")

    def check(self, row: dict[str, Any], result: dict[str, Any]) -> tuple[str | None, str, Any]:
        """(failure reason or None, detail, parsed JSON or None).  Reasons are reported even when
        not in retry_on; the caller decides whether to retry."""
        text = result.get("response") or ""
        parsed = None
        if result.get("finish_reason") == "length":
            return "length", "hit max_tokens", None
        if not text.strip():
            return "empty", "empty response", None
        if self.json_output or self.schema is not None:
            try:
                parsed = parse_json_output(text)
            except json.JSONDecodeError as e:
                return "invalid-json", f"not valid JSON: {e}", None
            if self.schema is not None:
                import jsonschema

                try:
                    jsonschema.validate(parsed, self.schema)
                except jsonschema.ValidationError as e:
                    return "schema", f"schema violation at {list(e.absolute_path)}: {e.message}", parsed
        if self.custom is not None:
            msg = self.custom(row, {**result, "parsed": parsed})
            if msg:
                return "validator", str(msg), parsed
        return None, "", parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "retry_on": sorted(self.retry_on),
            "max_retries": self.max_retries,
            "json_output": self.json_output,
            "schema": self.schema,
            "custom": getattr(self.custom, "__qualname__", None) if self.custom else None,
            "corrective": self.corrective,
            "max_tokens_cap": self.max_tokens_cap,
        }


def _corrective_turn(result: dict[str, Any], detail: str) -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": result.get("response") or ""},
        {
            "role": "user",
            "content": f"Your previous answer was rejected: {detail}. "
            "Fix exactly that problem and answer again in the required format only.",
        },
    ]


# ------------------------------------------------------------------ one work item


def _is_transport_error(e: Exception) -> bool:
    return isinstance(e, openai.APIConnectionError | openai.APITimeoutError | httpx.TransportError) or (
        isinstance(e, openai.APIStatusError) and e.status_code >= 500
    )


def _generate_item(
    *,
    pool: ServerPool,
    clients: list[OpenAI],
    server: int,
    idx: int,
    sample: int,
    n_samples: int,
    row: dict[str, Any],
    model: str,
    params: SamplingParams,
    retries: int,
    validation: Validation,
    global_system: str | None,
    log: Callable[[str], None],
) -> dict[str, Any]:
    out = {INDEX_KEY: idx, **({SAMPLE_KEY: sample} if n_samples > 1 else {}), **row}
    base_messages = row_messages(row, global_system)
    messages = list(base_messages)
    p = SamplingParams(**{**params.__dict__, "extra_body": dict(params.extra_body)})
    if params.seed is not None and n_samples > 1:
        p.seed = params.seed + sample
    cap = validation.max_tokens_cap or params.max_tokens * 4
    attempts = transport_failures = quality_retries = 0
    reasons: list[str] = []
    result: dict[str, Any] | None = None
    err: str | None = None
    while True:
        attempts += 1
        gen = pool.acquire(server)
        try:
            result = chat(clients[server], model=model, messages=messages, params=p)
        except ServerDead:
            raise
        except Exception as e:  # noqa: BLE001 - classified below
            pool.release(progressed=False)
            if _is_transport_error(e) and pool.on_transport_error(server, gen):
                attempts -= 1  # outage, not this row's fault
                continue
            transport_failures += 1
            err = f"{type(e).__name__}: {e}"
            log(f"[row {idx}] attempt {transport_failures}/{retries} failed: {err}")
            if transport_failures >= retries:
                break
            time.sleep(min(60, 2**transport_failures))
            continue
        pool.release(progressed=True)
        reason, detail, parsed = validation.check(row, result)
        if reason is None:
            err = None
            break
        if reason not in validation.retry_on:
            err = None if reason in ("length", "empty") else f"{reason}: {detail}"
            if reason not in ("length", "empty"):
                reasons.append(reason)
            break
        reasons.append(reason)
        if quality_retries >= validation.max_retries:
            err = f"{reason}: {detail} (after {quality_retries} re-asks)"
            break
        quality_retries += 1
        if reason == "length":
            p.max_tokens = min(p.max_tokens * 2, cap)
        elif validation.corrective and reason in ("invalid-json", "schema", "validator"):
            messages = [*base_messages, *_corrective_turn(result, detail)]
        else:
            messages = list(base_messages)
    if result is not None:
        out.update(
            responses=result["response"],
            reasoning=result["reasoning"],
            finish_reason=result["finish_reason"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        )
        if validation.json_output or validation.schema is not None:
            try:
                out["parsed"] = parse_json_output(result["response"] or "")
            except json.JSONDecodeError:
                out["parsed"] = None
    else:
        out.update(responses=None, reasoning=None, finish_reason=None)
    out.update(attempts=attempts, retry_reasons=reasons, error=err)
    return out


# ------------------------------------------------------------------ output files


def _is_success(row: dict[str, Any]) -> bool:
    return not row.get("error") and row.get("responses") is not None


def _key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row[INDEX_KEY]), int(row.get(SAMPLE_KEY, 0))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # torn last line from a killed run
    return rows


def completed_keys(paths: list[Path]) -> set[tuple[int, int]]:
    """(row index, sample) pairs with a successful output in any of ``paths``."""
    return {_key(r) for p in paths for r in read_jsonl_rows(p) if INDEX_KEY in r and _is_success(r)}


def completed_indices(path: Path) -> set[int]:
    """Row indices with a successful (first-sample) output in a JSONL file."""
    return {i for i, s in completed_keys([path]) if s == 0}


def merge_jsonl(sources: list[Path], dest: Path) -> list[dict[str, Any]]:
    """Best row per (index, sample) across ``sources`` (success wins), sorted, written to ``dest``."""
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for src in sources:
        for r in read_jsonl_rows(src):
            if INDEX_KEY not in r:
                continue
            k = _key(r)
            if k not in best or (_is_success(r) and not _is_success(best[k])):
                best[k] = r
    rows = [best[k] for k in sorted(best)]
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(dest)
    return rows


def compact_jsonl(path: Path) -> None:
    """Rewrite a JSONL output sorted by (row index, sample), one row per key (success wins)."""
    merge_jsonl([path], path)


# ------------------------------------------------------------------ batch runner


@dataclass
class BatchStats:
    submitted: int = 0
    wall_seconds: float = 0.0
    restarts: int = 0
    aborted: str | None = None
    completion_tokens: int = 0  # generated by this run's requests


def run_batch(
    *,
    client: OpenAI | list[OpenAI],
    rows: list[dict[str, Any]],
    model: str,
    params: SamplingParams,
    concurrency: int = 64,
    retries: int = 3,
    global_system: str | None = None,
    jsonl_path: Path | None = None,
    log: Callable[[str], None] = print,
    use_tqdm: bool = True,
    n_samples: int = 1,
    validation: Validation | None = None,
    pool: ServerPool | None = None,
    done_from: list[Path] | None = None,
    assign: Callable[[list[tuple[int, int]]], list[tuple[int, int]]] | None = None,
    stats: BatchStats | None = None,
) -> list[dict[str, Any]]:
    """Generate ``n_samples`` outputs for every row concurrently (round-robin over ``client``
    when it is a list, one client per server).

    With ``jsonl_path`` each finished item is appended (fsync'd) as it completes; items already
    successful in ``jsonl_path`` or any ``done_from`` file are skipped (resume); the file is
    compacted into order at the end.  ``assign`` picks this worker's share of the remaining
    (row, sample) keys (multi-node sharding).  ``pool`` supplies health tracking / recovery;
    without it, servers are assumed healthy.  Returns this file's rows (or all results)."""
    clients = client if isinstance(client, list) else [client]
    validation = validation or Validation()
    stats = stats if stats is not None else BatchStats()
    pool = pool or ServerPool([str(getattr(c, "base_url", f"client-{i}")) for i, c in enumerate(clients)], log=log)
    sources = [p for p in [jsonl_path, *(done_from or [])] if p is not None]
    done = completed_keys(sources) if sources else set()
    todo = [(i, s) for i in range(len(rows)) for s in range(n_samples) if (i, s) not in done]
    if assign is not None:
        todo = assign(todo)
    total = len(rows) * n_samples
    if done:
        log(f"resume: {len(done)}/{total} outputs already done")
    results: dict[tuple[int, int], dict[str, Any]] = {}
    lock = threading.Lock()
    fh = jsonl_path.open("a", encoding="utf-8") if jsonl_path else None
    n_err = 0
    pbar = tqdm(total=total, initial=total - len(todo), desc="generate", unit="out", disable=not use_tqdm)
    t0 = time.time()
    stats.submitted = len(todo)
    pool.start_monitor()
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futs = [
                ex.submit(
                    _generate_item,
                    pool=pool,
                    clients=clients,
                    server=k % len(clients),
                    idx=i,
                    sample=s,
                    n_samples=n_samples,
                    row=rows[i],
                    model=model,
                    params=params,
                    retries=retries,
                    validation=validation,
                    global_system=global_system,
                    log=log,
                )
                for k, (i, s) in enumerate(todo)
            ]
            for fut in as_completed(futs):
                try:
                    out = fut.result()
                except CancelledError:
                    continue  # not started before the batch was stopped; stays to-do for a rerun
                except ServerDead as e:
                    if stats.aborted is None:
                        stats.aborted = str(e)
                        log(f"[watchdog] stopping: {e}; completed outputs are kept, rerun to resume")
                        for f in futs:
                            f.cancel()
                    continue
                with lock:
                    results[_key(out)] = out
                    n_err += not _is_success(out)
                    stats.completion_tokens += out.get("completion_tokens") or 0
                    if fh:
                        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                    pbar.update(1)
                    pbar.set_postfix(errors=n_err, refresh=False)
    finally:
        pool.stop_monitor()
        pbar.close()
        stats.wall_seconds = time.time() - t0
        stats.restarts = pool.restarts
        if fh:
            fh.close()
            compact_jsonl(jsonl_path)
    if n_err:
        log(f"{n_err} outputs failed (kept with an 'error' field; rerun to retry them)")
    if jsonl_path:
        return read_jsonl_rows(jsonl_path)
    return [results[k] for k in sorted(results)]


# ------------------------------------------------------------------ run summary


def summarize(rows: list[dict[str, Any]], *, stats: BatchStats | None = None) -> dict[str, Any]:
    """Counts and token totals for an output file's rows (plus this run's timing if given)."""
    ok = [r for r in rows if _is_success(r)]
    ctoks = sum(r.get("completion_tokens") or 0 for r in rows)
    ptoks = sum(r.get("prompt_tokens") or 0 for r in rows)
    errors = Counter(str(r["error"]).split(":", 1)[0] for r in rows if r.get("error"))
    summary: dict[str, Any] = {
        "outputs": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "errors_by_type": dict(errors),
        "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in rows)),
        "truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
        "truncated_no_answer": sum(
            1 for r in ok if r.get("finish_reason") == "length" and not str(r.get("responses") or "").strip()
        ),
        "empty": sum(1 for r in ok if r.get("finish_reason") != "length" and not str(r.get("responses") or "").strip()),
        "re_asked": sum(1 for r in rows if r.get("retry_reasons")),
        "re_ask_reasons": dict(Counter(x for r in rows for x in (r.get("retry_reasons") or []))),
        "prompt_tokens": ptoks,
        "completion_tokens": ctoks,
        "mean_completion_tokens": round(ctoks / len(rows), 1) if rows else 0,
    }
    if stats is not None:
        summary["this_run"] = {
            "submitted": stats.submitted,
            "wall_seconds": round(stats.wall_seconds, 1),
            "server_restarts": stats.restarts,
            "aborted": stats.aborted,
            "completion_tokens": stats.completion_tokens,
            "completion_tok_per_s": round(stats.completion_tokens / stats.wall_seconds, 1) if stats.wall_seconds else 0,
        }
    return summary


def format_summary(s: dict[str, Any]) -> str:
    lines = [
        f"outputs {s['outputs']}: ok {s['ok']}, failed {s['failed']}"
        + (f" {s['errors_by_type']}" if s["errors_by_type"] else ""),
        f"truncated (finish=length) {s['truncated']} (of which no answer at all {s['truncated_no_answer']}), "
        f"empty {s['empty']}, re-asked {s['re_asked']}" + (f" {s['re_ask_reasons']}" if s["re_ask_reasons"] else ""),
        f"tokens: prompt {s['prompt_tokens']:,}, completion {s['completion_tokens']:,} "
        f"(mean {s['mean_completion_tokens']} per output)",
    ]
    run = s.get("this_run")
    if run and "nodes" in run:
        lines.append(
            f"this run: {len(run['nodes'])} nodes in {run['wall_seconds']:.0f}s, "
            f"worker exit codes {run['worker_exit_codes']}"
        )
    elif run:
        lines.append(
            f"this run: {run['submitted']} outputs in {run['wall_seconds']:.0f}s, "
            f"{run['completion_tokens']:,} completion tokens ({run['completion_tok_per_s']:,.0f} tok/s), "
            f"server restarts {run['server_restarts']}" + (f", ABORTED: {run['aborted']}" if run["aborted"] else "")
        )
    return "\n".join(lines)
