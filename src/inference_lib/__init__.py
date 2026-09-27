"""Standalone vLLM serving + batch generation for 8xH100 nodes."""

from __future__ import annotations

__version__ = "0.3.0"

from .client import (
    SamplingParams,
    build_messages,
    chat,
    compact_jsonl,
    completed_indices,
    generate_one,
    get_client,
    get_prompt,
    get_system,
    resolve_model_id,
    run_batch,
)
from .io import (
    SUPPORTED_INPUT_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
    detect_format,
    load_input,
    preview_rows,
    validate_rows,
    write_output,
)
from .registry import REGISTRY, ModelSpec, build_vllm_args, list_models, load_preset, resolve_model, resolve_model_path
from .server import (
    LaunchPlan,
    VllmServer,
    list_served_model_ids,
    serve,
    server_ready,
    server_serves_model,
    stop_containers,
)

__all__ = [
    "REGISTRY",
    "LaunchPlan",
    "ModelSpec",
    "SamplingParams",
    "SUPPORTED_INPUT_FORMATS",
    "SUPPORTED_OUTPUT_FORMATS",
    "VllmServer",
    "build_messages",
    "build_vllm_args",
    "chat",
    "compact_jsonl",
    "completed_indices",
    "detect_format",
    "generate_one",
    "get_client",
    "get_prompt",
    "get_system",
    "list_models",
    "list_served_model_ids",
    "load_input",
    "load_preset",
    "preview_rows",
    "resolve_model",
    "resolve_model_id",
    "resolve_model_path",
    "run_batch",
    "serve",
    "server_ready",
    "server_serves_model",
    "stop_containers",
    "validate_rows",
    "write_output",
]
