import pytest

from inference_lib.registry import (
    CONFIG_DIR,
    REGISTRY,
    build_vllm_args,
    generic_spec,
    load_preset,
    resolve_model,
    resolve_model_path,
)

EXPECTED = {
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "qwen3.5-9b",
    "qwen3.5-35b-a3b",
    "gemma4-26b-a4b-it",
    "gemma4-31b-it",
    "olmo3-32b-think-dpo",
    "deepseek-r1-distill-llama-8b",
    "deepseek-v4-flash",
    "glm-5.3-flash",
}


def test_every_preset_loads_and_is_named_after_its_file():
    for path in CONFIG_DIR.glob("*.yaml"):
        assert load_preset(path).key == path.stem
    assert set(REGISTRY) >= EXPECTED


def test_presets_fit_one_node():
    for spec in REGISTRY.values():
        gpus = spec.recommended_gpus or spec.tensor_parallel_size
        assert gpus <= 8 and gpus % spec.tensor_parallel_size == 0, spec.key


def test_lookup_by_repo_id_is_case_insensitive():
    assert resolve_model("qwen/qwen3-8b").key == "qwen3-8b"
    assert resolve_model("nope/nope") is None


def test_unknown_preset_field_rejected(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("hf_repo: a/b\nbogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        load_preset(p)


def test_dp_from_gpu_count():
    spec = REGISTRY["qwen3-8b"]
    args = build_vllm_args(spec, "/m", num_gpus=8)
    assert args[args.index("--data-parallel-size") + 1] == "8"
    assert "--tensor-parallel-size" not in args
    with pytest.raises(ValueError):
        build_vllm_args(REGISTRY["gemma4-31b-it"], "/m", num_gpus=3)


def test_expert_parallel_only_when_multi_gpu():
    spec = REGISTRY["glm-5.3-flash"]
    assert "--enable-expert-parallel" in build_vllm_args(spec, "/m", num_gpus=8)
    assert "--enable-expert-parallel" not in build_vllm_args(spec, "/m", num_gpus=1, tensor_parallel_size=1)


def test_parsers_and_served_name():
    args = build_vllm_args(REGISTRY["deepseek-v4-flash"], "/m", port=9000)
    assert args[:3] == ["/m", "--served-model-name", "deepseek-v4-flash"]
    for flag in (
        "--tokenizer-mode",
        "--reasoning-parser",
        "--tool-call-parser",
        "--enable-auto-tool-choice",
        "--trust-remote-code",
    ):
        assert flag in args
    assert args[args.index("--port") + 1] == "9000"


def test_generic_model_uses_vllm_defaults():
    spec = generic_spec("org/some-model")
    args = build_vllm_args(spec, "org/some-model")
    assert "--tensor-parallel-size" not in args and "--reasoning-parser" not in args


def test_resolve_path_prefers_models_dir(tmp_path, monkeypatch):
    d = tmp_path / "qwen3-8b"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_text("")
    monkeypatch.setenv("INFERENCE_MODELS_DIR", str(tmp_path))
    assert resolve_model_path("qwen3-8b") == str(d.resolve())


def test_resolve_path_hf_cache_then_repo_id(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_MODELS_DIR", str(tmp_path / "none"))
    monkeypatch.setattr("inference_lib.registry.DEFAULT_MODELS_DIR", tmp_path / "none")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert resolve_model_path("qwen3-8b") == "Qwen/Qwen3-8B"
    snap = tmp_path / "hf" / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text("{}")
    assert resolve_model_path("qwen3-8b") == str(snap)


def test_presets_sharing_a_repo_share_weights(tmp_path, monkeypatch):
    d = tmp_path / "glm-5.3-flash"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors.index.json").write_text("{}")
    monkeypatch.setenv("INFERENCE_MODELS_DIR", str(tmp_path))
    assert resolve_model_path("glm-5.3-flash-legacy") == str(d.resolve())
