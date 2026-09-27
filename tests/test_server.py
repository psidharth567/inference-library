import json
import os

from inference_lib.registry import REGISTRY
from inference_lib.server import LaunchPlan, _mount_root, api_root, model_id_matches, prepare_tokenizer_override


def test_api_root():
    assert api_root("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000"
    assert api_root("http://host:8000/v1/") == "http://host:8000"


def test_model_id_matches():
    assert model_id_matches("qwen3-8b", "qwen3-8b")
    assert model_id_matches("/w/Qwen3-8B", "Qwen/Qwen3-8B")
    assert not model_id_matches("qwen3-8b", "qwen3-14b")


def test_mount_root_for_hf_snapshot(tmp_path):
    snap = tmp_path / "models--a--b" / "snapshots" / "123"
    snap.mkdir(parents=True)
    assert _mount_root(str(snap)) == (tmp_path / "models--a--b").resolve()
    assert _mount_root("org/repo") is None


def test_docker_command(tmp_path, monkeypatch):
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_text("")
    plan = LaunchPlan(str(model), port=8123, gpus="2,3", docker=True, image="img:tag")
    cmd = plan.command()
    assert cmd[:3] == ["docker", "run", "-d"]
    assert '"device=2,3"' in cmd
    assert "--ipc=host" not in cmd  # see server.py docstring (logind RemoveIPC)
    assert f"{model.resolve()}:{model.resolve()}:ro" in cmd
    assert "LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64" in cmd
    assert f"/tmp/inference-lib-{os.getuid()}/{plan.name}:/home/inference" in cmd  # node-local, per server
    i = cmd.index("img:tag")
    assert cmd[i + 1] == str(model.resolve())
    assert "--data-parallel-size" in cmd  # 2 GPUs, generic TP=1 -> DP=2


def test_bare_metal_command(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_VLLM_BIN", "/opt/vllm")
    plan = LaunchPlan("org/model", port=8000, gpus=[0], docker=False)
    assert plan.command()[:3] == ["/opt/vllm", "serve", "org/model"]


def test_tokenizer_override(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer.json").write_text("{}")
    (src / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "LlamaTokenizerFast", "bos_token": "x"}))
    spec = REGISTRY["deepseek-r1-distill-llama-8b"]
    out = prepare_tokenizer_override(spec, str(src), tmp_path / "tok")
    cfg = json.loads((out / "tokenizer_config.json").read_text())
    assert cfg == {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "x"}
    assert (out / "tokenizer.json").exists()
    assert prepare_tokenizer_override(REGISTRY["qwen3-8b"], str(src), tmp_path / "tok") is None


def test_deployment_splits_fixed_size_servers(tmp_path, monkeypatch):
    from inference_lib.registry import ModelSpec
    from inference_lib.server import Deployment

    spec = ModelSpec(
        key="moe", hf_repo="org/moe", tensor_parallel_size=2, data_parallel_size=1, enable_expert_parallel=True
    )
    monkeypatch.setattr("inference_lib.server.resolve_model", lambda m: spec)
    dep = Deployment("org/moe", port=9000, gpus="0,1,2,3,4,5,6,7", image="img")
    assert [p.gpus for p in dep.plans] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert dep.base_urls == [f"http://127.0.0.1:{9000 + i}/v1" for i in range(4)]
    args = dep.plans[0].args
    assert args[args.index("--tensor-parallel-size") + 1] == "2" and "--data-parallel-size" not in args


def test_deployment_single_server_uses_internal_dp(monkeypatch):
    from inference_lib.server import Deployment

    dep = Deployment("qwen3-8b", port=9000, gpus="0,1,2,3", image="img")
    assert len(dep.plans) == 1
    args = dep.plans[0].args
    assert args[args.index("--data-parallel-size") + 1] == "4"
