"""OpenAI-compatible client helpers and the concurrent batch runner."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from tqdm import tqdm

PROMPT_KEYS = ("prompts", "prompt", "text", "input", "question", "query", "instruction", "content")
SYSTEM_KEYS = ("system", "system_prompt", "system_message", "sys")
INDEX_KEY = "_idx"


def get_prompt(row: dict[str, Any]) -> str:
    for k in PROMPT_KEYS:
        if row.get(k) is not None and str(row[k]).strip():
            return str(row[k])
    raise KeyError(f"each row needs one of {PROMPT_KEYS}; got keys {list(row)[:6]}")


def get_system(row: dict[str, Any], global_system: str | None = None) -> str | None:
    for k in SYSTEM_KEYS:
        if row.get(k) is not None and str(row[k]).strip():
            return str(row[k]).strip()
    return global_system.strip() if global_system and global_system.strip() else None


def build_messages(*, system: str | None, prompt: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return messages


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
    extra_body: dict[str, Any] = field(default_factory=dict)

    def request_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"max_tokens": self.max_tokens, "temperature": self.temperature}
        if self.top_p is not None:
            kw["top_p"] = self.top_p
        if self.seed is not None:
            kw["seed"] = self.seed
        extra = dict(self.extra_body)
        if self.top_k is not None:
            extra["top_k"] = self.top_k
        if self.enable_thinking is not None:
            extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = self.enable_thinking
        if extra:
            kw["extra_body"] = extra
        return kw


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


def _generate_row(client, idx, row, model, params, retries, global_system, log) -> dict[str, Any]:
    out = {INDEX_KEY: idx, **row}
    messages = build_messages(system=get_system(row, global_system), prompt=get_prompt(row))
    err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = chat(client, model=model, messages=messages, params=params)
            out.update(
                responses=r["response"],
                reasoning=r["reasoning"],
                finish_reason=r["finish_reason"],
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                error=None,
            )
            return out
        except Exception as e:  # noqa: BLE001 - any API failure is retried then recorded
            err = e
            log(f"[row {idx}] attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(min(60, 2**attempt))
    out.update(responses=None, reasoning=None, error=str(err))
    return out


def _is_success(row: dict[str, Any]) -> bool:
    return not row.get("error") and row.get("responses") is not None


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


def completed_indices(path: Path) -> set[int]:
    """Row indices with a successful response already in a JSONL output file."""
    return {r[INDEX_KEY] for r in read_jsonl_rows(path) if INDEX_KEY in r and _is_success(r)}


def compact_jsonl(path: Path) -> None:
    """Rewrite a JSONL output sorted by row index, one row per index (success wins)."""
    best: dict[int, dict[str, Any]] = {}
    for r in read_jsonl_rows(path):
        i = r.get(INDEX_KEY)
        if i is None:
            continue
        if i not in best or (_is_success(r) and not _is_success(best[i])):
            best[i] = r
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for i in sorted(best):
            f.write(json.dumps(best[i], ensure_ascii=False) + "\n")
    tmp.replace(path)


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
) -> list[dict[str, Any]]:
    """Generate for every row concurrently (round-robin over ``client`` if it is a list
    of clients, one per server replica).

    With ``jsonl_path`` each finished row is appended (fsync'd) as it completes, rows
    already successful in that file are skipped (resume), and the file is compacted
    into row order at the end.  Returns all output rows in input order.
    """
    clients = client if isinstance(client, list) else [client]
    done = completed_indices(jsonl_path) if jsonl_path else set()
    todo = [i for i in range(len(rows)) if i not in done]
    if done:
        log(f"resume: {len(done)}/{len(rows)} rows already done in {jsonl_path}")
    results: dict[int, dict[str, Any]] = {}
    lock = threading.Lock()
    fh = jsonl_path.open("a", encoding="utf-8") if jsonl_path else None
    n_err = 0
    pbar = tqdm(total=len(rows), initial=len(done), desc="generate", unit="row", disable=not use_tqdm)
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futs = [
                pool.submit(
                    _generate_row, clients[k % len(clients)], i, rows[i], model, params, retries, global_system, log
                )
                for k, i in enumerate(todo)
            ]
            for fut in as_completed(futs):
                out = fut.result()
                with lock:
                    results[out[INDEX_KEY]] = out
                    n_err += not _is_success(out)
                    if fh:
                        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                    pbar.update(1)
                    pbar.set_postfix(errors=n_err, refresh=False)
    finally:
        pbar.close()
        if fh:
            fh.close()
            compact_jsonl(jsonl_path)
    if n_err:
        log(f"{n_err} rows failed after {retries} attempts (kept with an 'error' field; rerun to retry them)")
    if jsonl_path:
        return read_jsonl_rows(jsonl_path)
    return [results[i] for i in sorted(results)]
