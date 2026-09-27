"""``inference`` command line: list / serve / stop / status / chat / batch / bench."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .client import SamplingParams, build_messages, chat, get_client, resolve_model_id, run_batch
from .io import (
    SUPPORTED_INPUT_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
    detect_format,
    load_input,
    preview_rows,
    validate_rows,
    write_output,
)
from .registry import REGISTRY, generic_spec, resolve_model, resolve_model_path
from .server import (
    Deployment,
    _mount_root,
    list_served_model_ids,
    prepare_tokenizer_override,
    running_containers,
    server_ready,
    stop_containers,
)

DEFAULT_PORT = int(os.environ.get("INFERENCE_PORT", "8000"))


def _served_name(model: str) -> str:
    spec = resolve_model(model)
    return (spec.served_model_name or spec.key) if spec else model


def _system_prompt(args) -> str | None:
    if getattr(args, "system_prompt_file", None):
        return Path(args.system_prompt_file).read_text(encoding="utf-8").strip() or None
    sp = getattr(args, "system_prompt", None) or os.environ.get("INFERENCE_SYSTEM_PROMPT")
    return sp.strip() if sp and sp.strip() else None


def _add_launch_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("server launch")
    g.add_argument("--port", type=int, default=DEFAULT_PORT)
    g.add_argument("--gpus", default=None, help='GPU indices "0,1,2,3", "all", or omit to take free GPUs')
    g.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="GPUs to use; extra GPUs become data-parallel ranks or extra servers (see `inference list`). "
        "Default: the preset's recommended_gpus",
    )
    g.add_argument("--max-model-len", type=int, default=None)
    g.add_argument("--tensor-parallel-size", "-tp", type=int, default=None)
    g.add_argument("--gpu-memory-utilization", type=float, default=None)
    g.add_argument(
        "--vllm-arg",
        action="append",
        default=[],
        dest="vllm_args",
        metavar="ARG",
        help="extra vLLM flag, repeatable: --vllm-arg=--enforce-eager --vllm-arg=--seed=1",
    )
    g.add_argument("--image", default=None, help="Docker image (default: preset image, or $INFERENCE_IMAGE)")
    g.add_argument(
        "--no-docker", action="store_true", help="run `vllm` from PATH / $INFERENCE_VLLM_BIN instead of Docker"
    )
    g.add_argument("--serve-log-dir", type=Path, default=None, help="server logs directory (default: <lib>/logs)")
    g.add_argument("--serve-timeout", type=int, default=3600, help="seconds to wait for the server to become ready")


def _deployment(args, *, keep_server: bool = False, log=None) -> Deployment:
    num_gpus = args.num_gpus
    if num_gpus is None and args.gpus is None:
        spec = resolve_model(args.model)
        num_gpus = spec.recommended_gpus if spec else None
    return Deployment(
        args.model,
        port=args.port,
        gpus=args.gpus,
        num_gpus=num_gpus,
        docker=not args.no_docker,
        image=args.image,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_args=args.vllm_args,
        log_dir=args.serve_log_dir,
        startup_timeout=args.serve_timeout,
        keep_server=keep_server,
        log=log,
    )


def _servers_for(args, log) -> tuple[Deployment | None, list[str]]:
    """Use --base-url (comma-separated for several replicas) or the server at --port if up;
    otherwise start the model's deployment (unless --no-auto-serve)."""
    urls = [u for u in (args.base_url or "").split(",") if u] or [f"http://127.0.0.1:{args.port}/v1"]
    if all(server_ready(u) for u in urls):
        if not args.base_url:  # auto-detect replicas on the following ports
            p = args.port + 1
            while server_ready(f"http://127.0.0.1:{p}/v1", timeout=2):
                urls.append(f"http://127.0.0.1:{p}/v1")
                p += 1
        return None, urls
    if args.no_auto_serve or args.base_url:
        raise SystemExit(f"no server at {', '.join(urls)}")
    dep = _deployment(args, keep_server=args.keep_server, log=log)
    dep.ensure_running()
    return dep, dep.base_urls


# ------------------------------------------------------------------ commands


def _layout(spec) -> str:
    gpus = spec.recommended_gpus or spec.tensor_parallel_size
    tp = spec.tensor_parallel_size
    ep = "+EP" if spec.enable_expert_parallel and gpus > 1 else ""
    if spec.data_parallel_size:
        per = tp * spec.data_parallel_size
        return f"{gpus // per} x (TP{tp}xDP{spec.data_parallel_size}{ep})"
    return f"TP{tp}xDP{gpus // tp}{ep}"


def cmd_list(args) -> int:
    print(f"{'KEY':<30} {'8-GPU LAYOUT':<18} {'HF REPO':<42} DESCRIPTION")
    for key in sorted(REGISTRY):
        s = REGISTRY[key]
        print(f"{key:<30} {_layout(s):<18} {s.hf_repo:<42} {s.description}")
    print("\nAny other HF repo id or local path also works (vLLM defaults). Presets: configs/models/*.yaml")
    return 0


def cmd_serve(args) -> int:
    dep = _deployment(args, keep_server=args.detach)
    if args.dry_run:
        for p in dep.plans:
            print(p.shell())
        return 0
    dep.ensure_running()
    for p in dep.plans:
        print(f"serving {p.served_model_name} at {p.base_url} on GPUs {p.gpus}", flush=True)
    if len(dep.plans) > 1:
        print(f"{len(dep.plans)} independent servers: `inference batch --port {args.port}` uses all of them")
    if args.detach:
        print(
            f"detached; stop with: inference stop --port {args.port}"
            + (" (and following ports)" if len(dep.plans) > 1 else "")
        )
        return 0
    try:
        while dep.alive():
            time.sleep(10)
        print("server exited", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        dep.shutdown()


def cmd_stop(args) -> int:
    names = running_containers()
    if args.port is not None:
        names = [n for n in names if int(n.rsplit("-", 1)[-1]) >= args.port]
    for n in stop_containers(names):
        print(f"stopped {n}")
    if not names:
        print("no inference containers running")
    return 0


def cmd_status(args) -> int:
    base_url = args.base_url or f"http://127.0.0.1:{args.port}/v1"
    if not server_ready(base_url):
        print(f"not ready: {base_url}")
        return 1
    print(f"ready: {base_url} serving {', '.join(list_served_model_ids(base_url))}")
    for n in running_containers():
        print(f"container: {n}")
    return 0


def _sampling(args) -> SamplingParams:
    thinking = None if args.thinking is None else args.thinking == "on"
    extra = json.loads(args.extra_body) if args.extra_body else {}
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        enable_thinking=thinking,
        extra_body=extra,
    )


def cmd_chat(args) -> int:
    prompt = args.prompt
    if args.input_file:
        prompt = Path(args.input_file).read_text(encoding="utf-8").strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("no prompt: use --prompt, --input-file or stdin", file=sys.stderr)
        return 1
    server, base_urls = _servers_for(args, log=lambda m: print(m, file=sys.stderr, flush=True))
    try:
        client = get_client(base_urls[0])
        model = resolve_model_id(client, _served_name(args.model))
        out = chat(
            client,
            model=model,
            messages=build_messages(system=_system_prompt(args), prompt=prompt),
            params=_sampling(args),
        )
        if args.show_reasoning and out["reasoning"]:
            print(f"--- reasoning ---\n{out['reasoning']}\n--- response ---")
        print(out["response"])
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({"prompt": prompt, "model": model, **out}, ensure_ascii=False, indent=2) + "\n"
            )
        return 0
    finally:
        if server:
            server.shutdown()


def cmd_batch(args) -> int:
    in_fmt = args.input_format or detect_format(args.input)
    out_fmt = args.output_format or detect_format(args.output)
    rows = load_input(args.input, in_fmt)
    ok, errs = validate_rows(rows)
    if not ok:
        print("input validation failed:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 1
    system = _system_prompt(args)
    print(
        f"{len(rows)} rows from {args.input} [{in_fmt}] -> {args.output} [{out_fmt}]\n{preview_rows(rows)}",
        file=sys.stderr,
    )
    if args.dry_run:
        return 0
    if out_fmt != "jsonl" and args.output.exists() and not args.overwrite:
        print(f"{args.output} exists; pass --overwrite", file=sys.stderr)
        return 1
    # every format streams through a JSONL (resumable); other formats are converted at the end
    jsonl = args.output if out_fmt == "jsonl" else args.output.with_name(args.output.name + ".partial.jsonl")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    server, base_urls = _servers_for(args, log)
    try:
        clients = [get_client(u) for u in base_urls]
        spec = resolve_model(args.model)
        concurrency = args.concurrency or (spec.default_concurrency(args.num_gpus) if spec else 256 * len(base_urls))
        model = resolve_model_id(clients[0], _served_name(args.model))
        log(f"generating with {model} on {len(base_urls)} server(s): {', '.join(base_urls)}")
        t0 = time.time()
        out_rows = run_batch(
            client=clients,
            rows=rows,
            model=model,
            params=_sampling(args),
            concurrency=concurrency,
            retries=args.retries,
            global_system=system,
            jsonl_path=jsonl,
            log=log,
            use_tqdm=not args.no_tqdm,
        )
        dt = time.time() - t0
        n_ok = sum(1 for r in out_rows if not r.get("error"))
        toks = sum(r.get("completion_tokens") or 0 for r in out_rows)
        log(f"{n_ok}/{len(rows)} rows ok in {dt:.0f}s ({toks / max(dt, 1e-9):.0f} completion tok/s this run)")
        if out_fmt != "jsonl":
            write_output(args.output, out_rows, out_fmt)
            jsonl.unlink(missing_ok=True)
        log(f"wrote {args.output}")
        return 0 if n_ok == len(rows) else 2
    finally:
        if server:
            server.shutdown()


def cmd_bench(args) -> int:
    """`vllm bench serve` (random dataset) against a running server, run from the serving image."""
    base_url = args.base_url or f"http://127.0.0.1:{args.port}/v1"
    if not server_ready(base_url):
        print(f"no server at {base_url}", file=sys.stderr)
        return 1
    served = list_served_model_ids(base_url)[0]
    name = args.model or served
    spec = resolve_model(name) or generic_spec(name)
    tok = resolve_model_path(name, resolve_model(name))
    override = prepare_tokenizer_override(spec, tok)
    image = args.image or os.environ.get("INFERENCE_IMAGE") or spec.image
    mount = []
    for path in {tok, str(override) if override else None} - {None}:
        root = _mount_root(path)
        if root is not None:
            mount += ["-v", f"{root}:{root}:ro"]
    tok = str(override) if override else tok
    root_url = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        *mount,
        "--entrypoint",
        "vllm",
        image,
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        root_url,
        "--model",
        served,
        "--tokenizer",
        tok,
        "--trust-remote-code",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--num-prompts",
        str(args.num_prompts or args.concurrency * 4),
        "--max-concurrency",
        str(args.concurrency),
        "--ignore-eos",
        "--seed",
        "0",
    ]
    return subprocess.call(cmd)


# ------------------------------------------------------------------ parser


def _add_gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        "-m",
        default=os.environ.get("INFERENCE_MODEL"),
        required=os.environ.get("INFERENCE_MODEL") is None,
        help="registry key (see `inference list`), HF repo id, or local path",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("INFERENCE_BASE_URL"),
        help="use these server(s), comma-separated "
        "(default: http://127.0.0.1:<port>/v1 plus replicas on the next ports)",
    )
    p.add_argument("--no-auto-serve", action="store_true", help="fail instead of starting a server")
    p.add_argument("--keep-server", action="store_true", help="leave an auto-started server running")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--system-prompt-file", type=Path, default=None)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--thinking",
        choices=["on", "off"],
        default=None,
        help="chat_template_kwargs.enable_thinking for hybrid-reasoning models (default: template default)",
    )
    p.add_argument("--extra-body", default=None, help="JSON merged into the request body, e.g. '{\"min_p\": 0.05}'")
    _add_launch_args(p)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inference", description="Standalone vLLM serving + batch generation (Docker, 8xH100)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list model presets").set_defaults(func=cmd_list)

    ps = sub.add_parser("serve", help="start a server and block until Ctrl+C")
    ps.add_argument("model", help="registry key, HF repo id, or local path")
    ps.add_argument("--dry-run", action="store_true", help="print the launch command and exit")
    ps.add_argument("--detach", action="store_true", help="leave the server running and return once ready")
    _add_launch_args(ps)
    ps.set_defaults(func=cmd_serve)

    pst = sub.add_parser("stop", help="stop inference containers on this host")
    pst.add_argument("--port", type=int, default=None, help="only containers serving this port and above")
    pst.set_defaults(func=cmd_stop)

    pss = sub.add_parser("status", help="probe a server")
    pss.add_argument("--port", type=int, default=DEFAULT_PORT)
    pss.add_argument("--base-url", default=None)
    pss.set_defaults(func=cmd_status)

    pc = sub.add_parser("chat", aliases=["generate"], help="one prompt")
    pc.add_argument("--prompt", "-p", default=None)
    pc.add_argument("--input-file", type=Path, default=None)
    pc.add_argument("--output", "-o", type=Path, default=None, help="also write the result as JSON")
    pc.add_argument("--show-reasoning", action="store_true")
    _add_gen_args(pc)
    pc.set_defaults(func=cmd_chat)

    pb = sub.add_parser("batch", help="generate for every row of a file (resumable)")
    pb.add_argument("--input", "-i", type=Path, required=True)
    pb.add_argument("--output", "-o", type=Path, required=True)
    pb.add_argument("--input-format", choices=SUPPORTED_INPUT_FORMATS, default=None)
    pb.add_argument("--output-format", choices=SUPPORTED_OUTPUT_FORMATS, default=None)
    pb.add_argument("--overwrite", action="store_true", help="replace an existing non-JSONL output")
    pb.add_argument("--dry-run", action="store_true", help="validate + preview the input only")
    pb.add_argument("--concurrency", type=int, default=None, help="in-flight requests (default: preset, ~256 per GPU)")
    pb.add_argument("--retries", type=int, default=3)
    pb.add_argument("--no-tqdm", action="store_true")
    _add_gen_args(pb)
    pb.set_defaults(func=cmd_batch)

    pbe = sub.add_parser("bench", help="throughput benchmark against a running server")
    pbe.add_argument("--port", type=int, default=DEFAULT_PORT)
    pbe.add_argument("--base-url", default=None)
    pbe.add_argument("--model", default=None, help="preset key (for the tokenizer); default: the served model")
    pbe.add_argument("--image", default=None)
    pbe.add_argument("--input-len", type=int, default=1024)
    pbe.add_argument("--output-len", type=int, default=1024)
    pbe.add_argument("--concurrency", type=int, default=256)
    pbe.add_argument("--num-prompts", type=int, default=None)
    pbe.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))
