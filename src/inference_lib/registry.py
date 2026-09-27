"""Model registry.

Every model preset lives in ``configs/models/<key>.yaml`` — that file is the single
source of truth (engine flags, parallelism, parsers, env).  This module only loads
those files and turns a preset into a ``vllm serve`` argument list.

Model weights are resolved without touching any other project:

1. an explicit directory path,
2. ``$INFERENCE_MODELS_DIR/<key>`` and ``<lib>/models/<key>`` (a symlink farm you own),
3. the HF cache under ``$HF_HOME`` (default ``<lib>/.cache/huggingface``),
4. otherwise the HF repo id itself (vLLM downloads it into ``$HF_HOME``).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LIB_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = LIB_ROOT / "configs" / "models"
DEFAULT_MODELS_DIR = LIB_ROOT / "models"
DEFAULT_HF_HOME = LIB_ROOT / ".cache" / "huggingface"
IMAGE_REPO = "ghcr.io/psidharth567/inference-library"
# vllm/vllm-openai:v0.30.0 + an OCI source label (identical layers); docker/vllm-v0.30.0/Dockerfile
DEFAULT_IMAGE = f"{IMAGE_REPO}:vllm-v0.30.0"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_repo: str
    description: str = ""
    # name clients send as `model`; default: the preset key
    served_model_name: str | None = None
    engine: str = "vllm"
    image: str = DEFAULT_IMAGE
    # Parallelism.  One server = tensor_parallel_size x data_parallel_size GPUs.
    # data_parallel_size unset: every GPU you give the model goes into ONE server as vLLM
    #   data-parallel ranks (independent replicas behind one endpoint; right for dense models).
    # data_parallel_size set: the GPUs are split into independent servers of that size on
    #   consecutive ports.  Use this for MoE: vLLM's in-server DP runs MoE layers in lockstep
    #   across DP ranks, which measured ~2x slower per GPU than separate TP=2 servers.
    tensor_parallel_size: int = 1
    data_parallel_size: int | None = None
    enable_expert_parallel: bool = False
    max_model_len: int = 32768
    gpu_memory_utilization: float | None = None  # None: vLLM default (0.92 in 0.30)
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    kv_cache_dtype: str | None = None
    attention_backend: str | None = None
    tokenizer_mode: str | None = None
    # replaces tokenizer_config.json's tokenizer_class (see server.prepare_tokenizer_override)
    tokenizer_class: str | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    trust_remote_code: bool = False
    vllm_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    # legacy images carry their own working env: don't add the launcher's defaults (CUDA compat path, timeouts)
    keep_image_env: bool = False
    # recommended GPU count on one 8xH100 node for max throughput
    recommended_gpus: int | None = None
    # in-flight requests `inference batch` keeps by default (None: 256 per GPU); servers cap
    # running sequences at max_num_seqs and queue the rest, so too high only costs latency
    batch_concurrency: int | None = None

    def default_concurrency(self, num_gpus: int | None = None) -> int:
        return self.batch_concurrency or 256 * (num_gpus or self.recommended_gpus or self.tensor_parallel_size)

    @property
    def gpus_per_server(self) -> int | None:
        """Fixed server size, or None when one server takes all GPUs given to it."""
        return self.tensor_parallel_size * self.data_parallel_size if self.data_parallel_size else None


def _spec_from_dict(data: dict[str, Any], source: Path | None = None) -> ModelSpec:
    allowed = {f.name for f in dataclasses.fields(ModelSpec)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{source or 'preset'}: unknown fields {sorted(unknown)}")
    data = dict(data)
    if "vllm_args" in data:
        data["vllm_args"] = tuple(str(a) for a in data["vllm_args"] or ())
    if "env" in data:
        data["env"] = {str(k): str(v) for k, v in (data["env"] or {}).items()}
    return ModelSpec(**data)


def load_preset(path: str | Path) -> ModelSpec:
    p = Path(path)
    data = yaml.safe_load(p.read_text()) or {}
    if "key" not in data:
        data["key"] = p.stem
    return _spec_from_dict(data, p)


def _load_registry() -> dict[str, ModelSpec]:
    reg: dict[str, ModelSpec] = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        spec = load_preset(path)
        reg[spec.key] = spec
    return reg


REGISTRY: dict[str, ModelSpec] = _load_registry()
_BY_REPO = {s.hf_repo.lower(): s for s in REGISTRY.values()}


def list_models() -> list[str]:
    return sorted(REGISTRY)


def resolve_model(key_or_repo: str) -> ModelSpec | None:
    """Preset for a registry key, HF repo id, or a YAML path; None for unknown models."""
    if key_or_repo in REGISTRY:
        return REGISTRY[key_or_repo]
    if key_or_repo.lower() in _BY_REPO:
        return _BY_REPO[key_or_repo.lower()]
    if key_or_repo.endswith((".yaml", ".yml")) and Path(key_or_repo).is_file():
        return load_preset(key_or_repo)
    return None


def generic_spec(model: str) -> ModelSpec:
    """Preset for a model that is not in the registry: vLLM defaults."""
    name = Path(model).name if Path(model).is_dir() else model
    return ModelSpec(key=name.replace("/", "--").lower(), hf_repo=model, description="generic (not in registry)")


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or DEFAULT_HF_HOME)


def _has_weights(d: Path) -> bool:
    if not (d / "config.json").exists():
        return False
    return any(d.glob("*.safetensors")) or any(d.glob("*.safetensors.index.json")) or any(d.glob("*.bin"))


def _hf_cache_snapshot(repo: str, hub: Path) -> Path | None:
    snaps = hub / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir(), key=lambda q: q.stat().st_mtime, reverse=True):
        if _has_weights(snap):
            return snap
    return None


def resolve_model_path(model: str, spec: ModelSpec | None = None) -> str:
    """Local weights directory for ``model`` if one exists, else the HF repo id."""
    p = Path(model).expanduser()
    if p.is_dir() and (p / "config.json").exists():
        return str(p.resolve())
    spec = spec or resolve_model(model)
    key = spec.key if spec else model
    repo = spec.hf_repo if spec else model
    dirs = [Path(d) for d in os.environ.get("INFERENCE_MODELS_DIR", "").split(":") if d]
    for base in [*dirs, DEFAULT_MODELS_DIR]:
        same_repo = [k for k, sp in REGISTRY.items() if sp.hf_repo.lower() == repo.lower() and k != key]
        for name in (key, *same_repo, repo.split("/")[-1], repo.replace("/", "--")):
            cand = base / name
            if cand.is_dir() and _has_weights(cand):
                return str(cand.resolve())
    home = hf_home()
    for hub in (home / "hub", home):
        snap = _hf_cache_snapshot(repo, hub)
        if snap is not None:
            return str(snap)
    return repo


def build_vllm_args(
    spec: ModelSpec,
    model_path: str,
    *,
    port: int = 8000,
    host: str = "0.0.0.0",
    num_gpus: int | None = None,
    served_model_name: str | None = None,
    max_model_len: int | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Arguments for ``vllm serve`` (without the leading ``vllm serve``) for ONE server
    on ``num_gpus`` GPUs (default: tp x preset data_parallel_size)."""
    tp = tensor_parallel_size or spec.tensor_parallel_size
    gpus = num_gpus or tp * (spec.data_parallel_size or 1)
    if gpus % tp:
        raise ValueError(f"{spec.key}: {gpus} GPUs is not a multiple of tensor_parallel_size={tp}")
    dp = gpus // tp
    args: list[str] = [model_path]
    args += ["--served-model-name", served_model_name or spec.served_model_name or spec.key]
    args += ["--host", host, "--port", str(port)]
    if tp > 1:
        args += ["--tensor-parallel-size", str(tp)]
    if dp > 1:
        args += ["--data-parallel-size", str(dp)]
    if spec.enable_expert_parallel and gpus > 1:
        args += ["--enable-expert-parallel"]
    args += ["--max-model-len", str(max_model_len or spec.max_model_len)]
    if gpu_memory_utilization or spec.gpu_memory_utilization:
        args += ["--gpu-memory-utilization", str(gpu_memory_utilization or spec.gpu_memory_utilization)]
    if spec.max_num_seqs:
        args += ["--max-num-seqs", str(spec.max_num_seqs)]
    if spec.max_num_batched_tokens:
        args += ["--max-num-batched-tokens", str(spec.max_num_batched_tokens)]
    if spec.kv_cache_dtype:
        args += ["--kv-cache-dtype", spec.kv_cache_dtype]
    if spec.attention_backend:
        args += ["--attention-backend", spec.attention_backend]
    if spec.tokenizer_mode:
        args += ["--tokenizer-mode", spec.tokenizer_mode]
    if spec.reasoning_parser:
        args += ["--reasoning-parser", spec.reasoning_parser]
    if spec.tool_call_parser:
        args += ["--tool-call-parser", spec.tool_call_parser, "--enable-auto-tool-choice"]
    if spec.trust_remote_code:
        args += ["--trust-remote-code"]
    args += list(spec.vllm_args)
    args += list(extra_args or [])
    return args
