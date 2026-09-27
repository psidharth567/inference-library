"""Reproducibility stamp: ``<output>.meta.json`` next to every batch output.

The stamp records what generated the file: the serving stack as observed on the live servers
(vLLM version, served models, image + image id of our containers), the preset file hash, the
sampling and validation settings, the input hash, hosts and the run summary.

Its ``fingerprint`` is the part that must not change while one output file is being filled.
Resuming into a file whose stored fingerprint differs (another image, vLLM version, preset,
sampling, ...) is refused unless explicitly allowed, so one file never silently mixes two stacks.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any

import httpx

from . import __version__
from .registry import CONFIG_DIR, LIB_ROOT, resolve_model
from .server import CONTAINER_PREFIX, api_root


def meta_path(output: Path) -> Path:
    return output.with_name(output.name + ".meta.json")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    r = subprocess.run(["git", "-C", str(LIB_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    dirty = subprocess.run(
        ["git", "-C", str(LIB_ROOT), "status", "--porcelain", "src", "configs"], capture_output=True, text=True
    ).stdout.strip()
    return r.stdout.strip() + ("-dirty" if dirty else "")


def _container_image(port: int) -> dict[str, str] | None:
    """Image of the local inference container serving ``port`` (name inference-<key>-<port>)."""
    r = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{CONTAINER_PREFIX}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    for name in r.stdout.split():
        if name.rsplit("-", 1)[-1] == str(port):
            q = subprocess.run(
                ["docker", "inspect", "-f", "{{.Config.Image}} {{.Image}}", name], capture_output=True, text=True
            )
            if q.returncode == 0 and q.stdout.strip():
                image, image_id = q.stdout.split()
                return {"container": name, "image": image, "image_id": image_id}
    return None


def observe_server(base_url: str) -> dict[str, Any]:
    root = api_root(base_url)
    info: dict[str, Any] = {"url": base_url, "observed_from": socket.gethostname()}
    try:
        info["vllm_version"] = httpx.get(root + "/version", timeout=10).json().get("version")
    except (httpx.HTTPError, ValueError):
        info["vllm_version"] = None
    try:
        data = httpx.get(root + "/v1/models", timeout=10).json().get("data", [])
        info["models"] = [{k: m.get(k) for k in ("id", "root", "max_model_len")} for m in data]
    except (httpx.HTTPError, ValueError):
        info["models"] = []
    host = urllib.parse.urlparse(base_url).hostname or ""
    if host in ("127.0.0.1", "localhost", socket.gethostname()):
        port = urllib.parse.urlparse(base_url).port or 80
        info["container"] = _container_image(port)
    return info


def preset_info(model: str) -> dict[str, Any]:
    spec = resolve_model(model)
    if spec is None:
        return {"model": model, "preset": None}
    path = CONFIG_DIR / f"{spec.key}.yaml"
    return {"model": model, "preset": spec.key, "preset_sha256": file_sha256(path) if path.exists() else None}


def fingerprint(
    servers: list[dict[str, Any]],
    preset: dict[str, Any],
    sampling: dict[str, Any],
    validation: dict[str, Any],
    n_samples: int,
    system_prompt: str | None,
) -> dict[str, Any]:
    versions = sorted({str(s.get("vllm_version")) for s in servers})
    models = sorted({json.dumps(m, sort_keys=True) for s in servers for m in s.get("models", [])})
    image_ids = sorted({s["container"]["image_id"] for s in servers if s.get("container")})
    return {
        "vllm_versions": versions,
        "served_models": [json.loads(m) for m in models],
        "image_ids": image_ids,
        "preset": preset.get("preset"),
        "preset_sha256": preset.get("preset_sha256"),
        "sampling": sampling,
        "validation": validation,
        "n_samples": n_samples,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest() if system_prompt else None,
    }


def fingerprint_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    keys = set(old) | set(new)
    return {k: {"stored": old.get(k), "now": new.get(k)} for k in sorted(keys) if old.get(k) != new.get(k)}


def check_resume(output: Path, new_fp: dict[str, Any], *, allow_mixed: bool) -> dict[str, Any] | None:
    """Stored stamp of an existing output, after checking its fingerprint matches ``new_fp``."""
    mp = meta_path(output)
    if not mp.exists():
        return None
    stored = json.loads(mp.read_text())
    diff = fingerprint_diff(stored.get("fingerprint", {}), new_fp)
    if diff and not allow_mixed:
        raise SystemExit(
            f"refusing to resume {output}: it was generated with a different setup than now:\n"
            + json.dumps(diff, indent=2)
            + "\nWrite to a new output file, or pass --allow-mixed to append anyway (recorded in the stamp)."
        )
    if diff:
        stored.setdefault("mixed_with", []).append({"at": time.time(), "diff": diff})
    return stored


def write_stamp(output: Path, stamp: dict[str, Any]) -> Path:
    mp = meta_path(output)
    tmp = mp.with_suffix(".tmp")
    tmp.write_text(json.dumps(stamp, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(mp)
    return mp


def build_stamp(
    *,
    input_path: Path,
    n_rows: int,
    servers: list[dict[str, Any]],
    preset: dict[str, Any],
    fp: dict[str, Any],
    previous: dict[str, Any] | None,
    run: dict[str, Any],
) -> dict[str, Any]:
    stamp = dict(previous or {})
    stamp.update(
        inference_lib_version=__version__,
        inference_lib_commit=_git_commit(),
        input={"path": str(input_path), "sha256": file_sha256(input_path), "rows": n_rows},
        preset=preset,
        servers=servers,
        fingerprint=fp,
    )
    stamp.setdefault("runs", []).append(run)
    return stamp
