"""Start / stop / probe an OpenAI-compatible vLLM server.

Default runtime is Docker (one pinned vLLM image for every model).  ``docker=False``
runs a ``vllm`` binary from ``$INFERENCE_VLLM_BIN`` or ``PATH`` instead.

Host notes baked in here (8xH100, driver 535 / CUDA 12.2):
- the image ships CUDA 13 user-space; ``/usr/local/cuda/compat`` provides the
  forward-compatible driver libs, so it goes first on ``LD_LIBRARY_PATH``;
- containers run as the calling user (no root-owned files on shared storage) with a
  private IPC namespace: with ``--ipc=host`` systemd-logind's RemoveIPC deletes the
  user's semaphores in /dev/shm when any ssh session of that user ends, which kills
  multi-GPU vLLM startup.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path

import httpx

from .registry import LIB_ROOT, ModelSpec, build_vllm_args, generic_spec, hf_home, resolve_model, resolve_model_path

TOKENIZER_DIR = LIB_ROOT / ".cache" / "tokenizers"  # tokenizer overrides, shared read-only
CONTAINER_HOME = "/home/inference"
CONTAINER_TOKENIZERS = "/opt/inference-tokenizers"


def cache_root() -> Path:
    """Node-local root for per-server HOME dirs (torch.compile / Triton / FlashInfer / DeepGEMM
    caches).  Kept off shared storage and never shared between servers: concurrent servers
    writing one cache produced corrupt Inductor entries and missing FlashInfer cubins."""
    return Path(os.environ.get("INFERENCE_CACHE_DIR") or f"/tmp/inference-lib-{os.getuid()}")


CONTAINER_PREFIX = "inference-"
BASE_ENV = {
    "LD_LIBRARY_PATH": "/usr/local/cuda/compat:/usr/local/cuda/lib64",
    # cold starts of the 150-300 GB models (weights from shared storage + kernel JIT) exceed the 600s default
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
}


# ---------------------------------------------------------------- probing


def api_root(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path or "", "", "", ""))


def list_served_model_ids(base_url: str, timeout: float = 5.0) -> list[str]:
    response = httpx.get(f"{api_root(base_url)}/v1/models", timeout=timeout)
    response.raise_for_status()
    return [str(item["id"]) for item in response.json().get("data", []) if "id" in item]


def server_ready(base_url: str, timeout: float = 5.0) -> bool:
    try:
        return httpx.get(f"{api_root(base_url)}/v1/models", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def model_id_matches(served_id: str, expected: str) -> bool:
    a, b = served_id.rstrip("/"), expected.rstrip("/")
    return a == b or a.split("/")[-1].lower() == b.split("/")[-1].lower()


def server_serves_model(base_url: str, expected: str, timeout: float = 5.0) -> bool:
    try:
        return any(model_id_matches(s, expected) for s in list_served_model_ids(base_url, timeout))
    except httpx.HTTPError:
        return False


# ---------------------------------------------------------------- GPUs


def free_gpus(max_used_mib: int = 1024) -> list[int]:
    """Indices of GPUs on this host with (almost) no memory in use."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    free = []
    for line in out.strip().splitlines():
        idx, used = (x.strip() for x in line.split(","))
        if int(used) <= max_used_mib:
            free.append(int(idx))
    return free


def pick_gpus(gpus: str | list[int] | None, count: int) -> list[int]:
    """``gpus`` is an explicit list / "0,1" string / "all"; otherwise take ``count`` free GPUs."""
    if isinstance(gpus, list):
        return gpus
    if gpus and gpus != "auto":
        if gpus == "all":
            return free_gpus(max_used_mib=10**9)
        return [int(g) for g in str(gpus).split(",") if g.strip()]
    free = free_gpus()
    if len(free) < count:
        raise RuntimeError(f"need {count} free GPUs, found {len(free)} free: {free}. Pass --gpus explicitly.")
    return free[:count]


# ---------------------------------------------------------------- launch plan


def _mount_root(path: str) -> Path | None:
    """Directory to bind-mount so ``path`` resolves inside the container (HF snapshots
    contain relative symlinks into ../../blobs, so mount the whole repo cache dir)."""
    p = Path(path)
    if not p.is_dir():
        return None
    real = p.resolve()
    parts = real.parts
    if "snapshots" in parts:
        return Path(*parts[: parts.index("snapshots")])
    return real


def prepare_tokenizer_override(spec: ModelSpec, model_path: str, root: Path = TOKENIZER_DIR) -> Path | None:
    """Copy of the model's tokenizer files with ``tokenizer_class`` replaced.

    Needed where transformers>=5 maps a declared class to the wrong pipeline, e.g.
    DeepSeek-R1-Distill-Llama declares LlamaTokenizerFast but ships a byte-level BPE
    tokenizer.json: transformers 5 rebuilds it as SentencePiece and both encode and
    decode are wrong.  PreTrainedTokenizerFast uses tokenizer.json verbatim.
    """
    cls = spec.tokenizer_class
    if not cls:
        return None
    src = Path(model_path)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(model_path, allow_patterns=["tokenizer*", "special_tokens_map.json", "*.jinja"]))
    digest = hashlib.sha1(f"{src.resolve()}:{cls}".encode()).hexdigest()[:10]
    dest = root / f"{spec.key}-{digest}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "special_tokens_map.json", "chat_template.jinja"):
        if (src / name).exists():
            shutil.copyfile(src / name, dest / name)
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    cfg["tokenizer_class"] = cls
    (dest / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    return dest


class LaunchPlan:
    """Everything needed to start one server; ``command()`` is the exact argv."""

    def __init__(
        self,
        model: str,
        *,
        port: int = 8000,
        gpus: str | list[int] | None = None,
        num_gpus: int | None = None,
        docker: bool = True,
        image: str | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float | None = None,
        vllm_args: list[str] | None = None,
        served_model_name: str | None = None,
        name: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.spec = resolve_model(model) or generic_spec(model)
        self.model_path = resolve_model_path(model, self.spec if resolve_model(model) else None)
        tp = tensor_parallel_size or self.spec.tensor_parallel_size
        self.gpus = pick_gpus(gpus, num_gpus or tp)
        self.port = port
        self.docker = docker
        self.image = image or os.environ.get("INFERENCE_IMAGE") or self.spec.image
        self.served_model_name = served_model_name or self.spec.served_model_name or self.spec.key
        self.name = name or f"{CONTAINER_PREFIX}{self.spec.key}-{port}"
        self.home = cache_root() / self.name
        self.env = {**({} if self.spec.keep_image_env else BASE_ENV), **self.spec.env, **(env or {})}
        extra = list(vllm_args or [])
        tok = prepare_tokenizer_override(self.spec, self.model_path)
        if tok is not None and "--tokenizer" not in extra:
            extra += ["--tokenizer", f"{CONTAINER_TOKENIZERS}/{tok.name}" if docker else str(tok)]
        self.args = build_vllm_args(
            self.spec,
            self.model_path,
            port=port,
            num_gpus=len(self.gpus),
            served_model_name=self.served_model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            extra_args=extra,
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def command(self) -> list[str]:
        if not self.docker:
            vllm = os.environ.get("INFERENCE_VLLM_BIN") or shutil.which("vllm") or "vllm"
            return [vllm, "serve", *self.args]
        hf = hf_home()
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.name,
            "--gpus",
            f'"device={",".join(map(str, self.gpus))}"',
            "--network",
            "host",
            "--shm-size",
            "64g",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"HOME={CONTAINER_HOME}",
            "-e",
            f"HF_HOME={hf}",
            "-v",
            f"{self.home}:{CONTAINER_HOME}",
            "-v",
            f"{hf}:{hf}",
            "-v",
            f"{TOKENIZER_DIR}:{CONTAINER_TOKENIZERS}:ro",
            # the calling uid has no passwd entry in the image; older vLLM/torch call getpwuid()
            "-v",
            f"{self.home / '.passwd'}:/etc/passwd:ro",
            "-v",
            f"{self.home / '.group'}:/etc/group:ro",
        ]
        for k, v in self.env.items():
            cmd += ["-e", f"{k}={v}"]
        root = _mount_root(self.model_path)
        if root is not None:
            cmd += ["-v", f"{root}:{root}:ro"]
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if os.environ.get(k):
                cmd += ["-e", k]
        return [*cmd, self.image, *self.args]

    def shell(self) -> str:
        return " ".join(shlex.quote(c) for c in self.command())


# ---------------------------------------------------------------- lifecycle


def write_identity_files(image: str, home: Path) -> None:
    """``home/.passwd`` / ``home/.group``: the image's own files plus an entry for the calling
    uid/gid (HOME = the container home), mounted over /etc/passwd and /etc/group."""
    uid, gid = os.getuid(), os.getgid()
    entries = (
        ("passwd", "root:x:0:0:root:/root:/bin/bash\n", f"inference:x:{uid}:{gid}::{CONTAINER_HOME}:/bin/bash", uid),
        ("group", "root:x:0:\n", f"inference:x:{gid}:", gid),
    )
    for name, fallback, line, own_id in entries:
        r = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "cat", image, f"/etc/{name}"], capture_output=True, text=True
        )
        content = r.stdout if r.returncode == 0 and r.stdout.strip() else fallback
        ids = {f.split(":")[2] for f in content.splitlines() if f.count(":") >= 2}
        if str(own_id) not in ids:
            content = content.rstrip("\n") + "\n" + line + "\n"
        (home / f".{name}").write_text(content)


def _docker_state(name: str) -> str | None:
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}}", name], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


class VllmServer:
    """Owns one server process/container: start, wait for ready, stream logs, stop."""

    def __init__(
        self,
        plan: LaunchPlan,
        *,
        log_path: str | Path | None = None,
        startup_timeout: int = 3600,
        keep_server: bool = False,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.plan = plan
        self.base_url = plan.base_url
        self.log_path = Path(log_path) if log_path else LIB_ROOT / "logs" / f"{plan.name}.log"
        self.startup_timeout = startup_timeout
        self.keep_server = keep_server
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.proc: subprocess.Popen | None = None
        self._log_proc: subprocess.Popen | None = None
        self._log_handle = None
        self.started_by_us = False
        self._stopped = False

    def ensure_running(self) -> None:
        if server_ready(self.base_url):
            if server_serves_model(self.base_url, self.plan.served_model_name):
                self.log(f"reusing server at {self.base_url} ({self.plan.served_model_name})")
                return
            served = ", ".join(list_served_model_ids(self.base_url)) or "(none)"
            raise RuntimeError(
                f"{self.base_url} is serving {served}, not {self.plan.served_model_name}; use another --port"
            )
        self.start()
        self.wait_ready()

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        for d in (self.plan.home, TOKENIZER_DIR, hf_home()):  # else docker creates them root-owned
            d.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self.log(f"GPUs {self.plan.gpus} | model {self.plan.model_path}")
        self.log(f"starting: {self.plan.shell()}")
        self.log(f"server log -> {self.log_path}")
        if self.plan.docker:
            if shutil.which("docker") is None:
                raise RuntimeError("docker not found; install it or pass --no-docker")
            subprocess.run(["docker", "rm", "-f", self.plan.name], capture_output=True)
            write_identity_files(self.plan.image, self.plan.home)
            r = subprocess.run(self.plan.command(), capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"docker run failed: {r.stderr.strip()}")
            self._log_proc = subprocess.Popen(
                ["docker", "logs", "-f", self.plan.name],
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        else:
            env = {**os.environ, **self.plan.env, "CUDA_VISIBLE_DEVICES": ",".join(map(str, self.plan.gpus))}
            env.setdefault("HF_HOME", str(hf_home()))
            self.proc = subprocess.Popen(
                self.plan.command(),
                env=env,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.started_by_us = True
        atexit.register(self.shutdown)

    def _alive(self) -> bool:
        if self.plan.docker:
            return _docker_state(self.plan.name) == "running"
        return self.proc is not None and self.proc.poll() is None

    def _tail(self, n: int = 25) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def wait_ready(self) -> None:
        t0 = time.time()
        last = 0.0
        while time.time() - t0 < self.startup_timeout:
            if server_ready(self.base_url):
                self.log(f"server ready at {self.base_url} after {time.time() - t0:.0f}s")
                return
            if self.started_by_us and not self._alive():
                raise RuntimeError(f"server exited during startup; last log lines:\n{self._tail()}")
            if time.time() - last >= 60:
                self.log(f"waiting for server... {time.time() - t0:.0f}s")
                last = time.time()
            time.sleep(5)
        raise TimeoutError(f"server not ready after {self.startup_timeout}s; see {self.log_path}")

    def shutdown(self) -> None:
        if self._stopped or not self.started_by_us or self.keep_server:
            return
        self._stopped = True
        self.log("stopping server")
        if self.plan.docker:
            subprocess.run(["docker", "rm", "-f", self.plan.name], capture_output=True, timeout=120)
        elif self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._log_proc is not None and self._log_proc.poll() is None:
            self._log_proc.terminate()
        if self._log_handle and not self._log_handle.closed:
            self._log_handle.close()


class Deployment:
    """One model on ``num_gpus`` GPUs: a single server, or several independent servers on
    consecutive ports when the preset fixes a server size (``data_parallel_size``)."""

    def __init__(
        self,
        model: str,
        *,
        port: int = 8000,
        gpus: str | list[int] | None = None,
        num_gpus: int | None = None,
        tensor_parallel_size: int | None = None,
        log_dir: str | Path | None = None,
        startup_timeout: int = 3600,
        keep_server: bool = False,
        log: Callable[[str], None] | None = None,
        **plan_kwargs,
    ) -> None:
        spec = resolve_model(model) or generic_spec(model)
        tp = tensor_parallel_size or spec.tensor_parallel_size
        per = tp * spec.data_parallel_size if spec.data_parallel_size else None
        explicit = pick_gpus(gpus, 0) if gpus not in (None, "auto") else None
        want = len(explicit) if explicit else (num_gpus or per or tp)
        all_gpus = explicit or pick_gpus(None, want)
        if per is None:
            groups = [all_gpus]
        else:
            if len(all_gpus) % per:
                raise ValueError(f"{spec.key}: servers use {per} GPUs each; {len(all_gpus)} GPUs is not a multiple")
            groups = [all_gpus[i : i + per] for i in range(0, len(all_gpus), per)]
        self.plans = [
            LaunchPlan(model, port=port + i, gpus=g, tensor_parallel_size=tensor_parallel_size, **plan_kwargs)
            for i, g in enumerate(groups)
        ]
        log_dir = Path(log_dir) if log_dir else None
        self.servers = [
            VllmServer(
                p,
                log_path=(log_dir / f"{p.name}.log") if log_dir else None,
                startup_timeout=startup_timeout,
                keep_server=keep_server,
                log=log,
            )
            for p in self.plans
        ]

    @property
    def base_urls(self) -> list[str]:
        return [p.base_url for p in self.plans]

    @property
    def served_model_name(self) -> str:
        return self.plans[0].served_model_name

    def ensure_running(self) -> None:
        todo = []
        for srv in self.servers:
            if server_ready(srv.base_url):
                srv.ensure_running()  # reuse, or raise if it serves another model
            else:
                srv.start()
                todo.append(srv)
        try:
            for srv in todo:
                srv.wait_ready()
        except BaseException:
            self.shutdown()
            raise

    def alive(self) -> bool:
        return any(server_ready(u, timeout=30) for u in self.base_urls)

    def shutdown(self) -> None:
        for srv in self.servers:
            srv.shutdown()


def running_containers() -> list[str]:
    r = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{CONTAINER_PREFIX}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    return [n for n in r.stdout.split() if n]


def stop_containers(names: list[str] | None = None) -> list[str]:
    names = names if names is not None else running_containers()
    for n in names:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True)
    return names


from contextlib import contextmanager  # noqa: E402


@contextmanager
def serve(model: str, **kwargs):
    """``with serve("qwen3-8b", num_gpus=8) as d: run_batch(clients=d.base_urls, ...)``"""
    dep = Deployment(model, **kwargs)
    dep.ensure_running()
    try:
        yield dep
    finally:
        dep.shutdown()
