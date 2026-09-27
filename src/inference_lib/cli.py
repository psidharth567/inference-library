"""``inference`` command line: list / serve / stop / status / chat / batch / bench."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .client import (
    BatchStats,
    SamplingParams,
    Validation,
    build_messages,
    chat,
    format_summary,
    get_client,
    json_schema_format,
    read_jsonl_rows,
    resolve_model_id,
    run_batch,
    summarize,
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
from .stamp import build_stamp, check_resume, fingerprint, meta_path, observe_server, preset_info, write_stamp
from .watchdog import ServerPool

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
    schema = _json_schema(args)
    if schema is not None:
        response_format = json_schema_format(schema)
    elif getattr(args, "json", False):
        response_format = {"type": "json_object"}
    else:
        response_format = None
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        enable_thinking=thinking,
        response_format=response_format,
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


def _load_validator(spec: str):
    """``path/to/file.py:func`` or ``package.module:func``; func(row, result) -> error or None."""
    target, func = spec.rsplit(":", 1)
    if target.endswith(".py"):
        mspec = importlib.util.spec_from_file_location("inference_user_validator", target)
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return getattr(mod, func)


def _json_schema(args) -> dict | None:
    return json.loads(Path(args.json_schema).read_text()) if getattr(args, "json_schema", None) else None


def _validation(args) -> Validation:
    schema = _json_schema(args)
    json_mode = bool(args.json) or schema is not None
    if args.retry_on is None:
        retry = (
            {"empty"}
            | ({"invalid-json", "schema"} if json_mode else set())
            | ({"validator"} if args.validator else set())
        )
    elif args.retry_on.strip() == "none":
        retry = set()
    else:
        retry = {x.strip() for x in args.retry_on.split(",") if x.strip()}
    return Validation(
        retry_on=frozenset(retry),
        max_retries=args.validation_retries,
        json_output=json_mode,
        schema=schema,
        custom=_load_validator(args.validator) if args.validator else None,
        corrective=not args.no_corrective,
        max_tokens_cap=args.max_tokens_cap,
    )


def _on_sigterm(signum, frame):  # let finally-blocks stop the servers we started
    raise SystemExit(128 + signum)


def cmd_batch(args) -> int:
    in_fmt = args.input_format or detect_format(args.input)
    out_fmt = args.output_format or detect_format(args.output)
    rows = load_input(args.input, in_fmt)
    ok, errs = validate_rows(rows)
    if not ok:
        print("input validation failed:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 1
    system = _system_prompt(args)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    log(f"{len(rows)} rows x {args.n} sample(s) from {args.input} [{in_fmt}] -> {args.output} [{out_fmt}]")
    log(preview_rows(rows))
    validation = _validation(args)
    sampling = _sampling(args)
    if args.dry_run:
        log(f"sampling {sampling.to_dict()}\nvalidation {validation.to_dict()}")
        return 0
    if out_fmt != "jsonl" and args.output.exists() and not args.overwrite and not args.shard_name:
        print(f"{args.output} exists; pass --overwrite", file=sys.stderr)
        return 1
    if args.nodes:
        if args.base_url:
            raise SystemExit("--nodes starts servers on each node; drop --base-url")
        from . import multinode

        return multinode.run(args, sys.argv[1:], n_rows=len(rows), log=log)

    signal.signal(signal.SIGTERM, _on_sigterm)
    output = args.output.resolve()
    final_jsonl = output if out_fmt == "jsonl" else output.with_name(output.name + ".partial.jsonl")
    assign = None
    done_from: list[Path] = []
    references: list[Path] = []  # stamps whose fingerprint this run must match
    if args.shard_name:  # worker of a multi-node batch
        from .multinode import shard_dir

        sdir = shard_dir(output)
        jsonl = sdir / f"{args.shard_name}.jsonl"
        assigned = {tuple(k) for k in json.loads(Path(args.assignment).read_text())[args.shard_name]}
        assign = lambda todo: [k for k in todo if k in assigned]  # noqa: E731
        done_from = [p for p in [final_jsonl, *sdir.glob("*.jsonl")] if p != jsonl and p.exists()]
        references = [output, *(p for p in sdir.glob("*.jsonl") if p != jsonl)]
        expected = len(assigned)
    else:
        jsonl = final_jsonl
        expected = len(rows) * args.n
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    dep, base_urls = _servers_for(args, log)
    try:
        clients = [get_client(u, timeout=args.request_timeout) for u in base_urls]
        pool = ServerPool(
            base_urls,
            recover=dep.restart if dep is not None else None,
            max_restarts=args.max_restarts,
            stall_timeout=args.stall_timeout,
            server_wait=args.server_wait,
            log=log,
        )
        spec = resolve_model(args.model)
        concurrency = args.concurrency or (spec.default_concurrency(args.num_gpus) if spec else 256 * len(base_urls))
        model = resolve_model_id(clients[0], _served_name(args.model))
        servers = [observe_server(u) for u in base_urls]
        preset = preset_info(args.model)
        fp = fingerprint(servers, preset, sampling.to_dict(), validation.to_dict(), args.n, system)
        for ref in references:
            check_resume(ref, fp, allow_mixed=args.allow_mixed)
        previous = check_resume(jsonl if args.shard_name else output, fp, allow_mixed=args.allow_mixed)
        stamp_target = jsonl if args.shard_name else output
        run_rec = {
            "host": socket.gethostname(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "argv": sys.argv[1:],
            "status": "running",
        }
        stamp = build_stamp(
            input_path=args.input,
            n_rows=len(rows),
            servers=servers,
            preset=preset,
            fp=fp,
            previous=previous,
            run=run_rec,
        )
        write_stamp(stamp_target, stamp)
        log(f"generating with {model} on {len(base_urls)} server(s): {', '.join(base_urls)}; concurrency {concurrency}")
        stats = BatchStats()
        out_rows = run_batch(
            client=clients,
            rows=rows,
            model=model,
            params=sampling,
            concurrency=concurrency,
            retries=args.retries,
            global_system=system,
            jsonl_path=jsonl,
            log=log,
            use_tqdm=not args.no_tqdm,
            n_samples=args.n,
            validation=validation,
            pool=pool,
            done_from=done_from,
            assign=assign,
            stats=stats,
        )
        summary = summarize(out_rows, stats=stats)
        run_rec.update(
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            status="aborted" if stats.aborted else "finished",
            summary=summary,
        )
        stamp["summary"] = summary
        write_stamp(stamp_target, stamp)
        log(format_summary(summary))
        if not args.shard_name and out_fmt != "jsonl":
            write_output(args.output, out_rows, out_fmt)
            final_jsonl.unlink(missing_ok=True)
        log(f"wrote {jsonl if args.shard_name else args.output}")
        return 0 if summary["ok"] >= expected and not stats.aborted else 2
    finally:
        if dep:
            dep.shutdown()


def cmd_summary(args) -> int:
    rows = read_jsonl_rows(args.file) if args.file.suffix == ".jsonl" else load_input(args.file)
    print(format_summary(summarize(rows)))
    mp = meta_path(args.file)
    if mp.exists():
        st = json.loads(mp.read_text())
        fp = st.get("fingerprint", {})
        print(
            f"stamp {mp.name}: vLLM {fp.get('vllm_versions')}, images {fp.get('image_ids')}, "
            f"preset {fp.get('preset')} ({str(fp.get('preset_sha256'))[:12]}), runs {len(st.get('runs', []))}"
            + (f", MIXED setups {len(st['mixed_with'])}x" if st.get("mixed_with") else "")
        )
    return 0


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
    p.add_argument(
        "--json", action="store_true", help="ask for a JSON object (server-side guided decoding) and parse it"
    )
    p.add_argument(
        "--json-schema",
        type=Path,
        default=None,
        help="JSON Schema file: guided decoding + validation of every answer (implies --json)",
    )
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
    pb.add_argument("--retries", type=int, default=3, help="attempts per output on request errors")
    pb.add_argument("--no-tqdm", action="store_true")
    pb.add_argument("--n", type=int, default=1, help="samples per row (seed+k per sample when --seed is set)")
    q = pb.add_argument_group("output validation / re-asking")
    q.add_argument(
        "--retry-on",
        default=None,
        help="comma list of empty,length,invalid-json,schema,validator or 'none' "
        "(default: empty, plus invalid-json,schema with --json/--json-schema, plus validator)",
    )
    q.add_argument("--validation-retries", type=int, default=2, help="max re-asks per output for --retry-on reasons")
    q.add_argument(
        "--no-corrective",
        action="store_true",
        help="re-ask with the original messages only (default: show the rejected answer + reason)",
    )
    q.add_argument(
        "--max-tokens-cap",
        type=int,
        default=None,
        help="on 'length' re-asks max_tokens doubles up to this (default 4x --max-tokens)",
    )
    q.add_argument(
        "--validator",
        default=None,
        help="file.py:func or module:func; func(row, result) returns an error string or None",
    )
    w = pb.add_argument_group("server watchdog")
    w.add_argument("--max-restarts", type=int, default=3, help="server restarts before the batch stops (resumable)")
    w.add_argument(
        "--stall-timeout",
        type=float,
        default=1800.0,
        help="restart servers when requests are in flight but none finished for this long (0: off)",
    )
    w.add_argument(
        "--server-wait",
        type=float,
        default=900.0,
        help="for servers this run did not start: how long to wait for them to come back",
    )
    w.add_argument("--request-timeout", type=float, default=3600.0, help="per-request HTTP timeout (seconds)")
    m = pb.add_argument_group("multi-node / reproducibility")
    m.add_argument(
        "--nodes", default=None, help="comma list of GPU hosts: serve the model on each and split the batch across them"
    )
    m.add_argument(
        "--allow-mixed",
        action="store_true",
        help="resume into an output generated with a different stack/settings (recorded in the stamp)",
    )
    m.add_argument("--assignment", default=None, help=argparse.SUPPRESS)
    m.add_argument("--shard-name", default=None, help=argparse.SUPPRESS)
    m.add_argument("--run-token", default=None, help=argparse.SUPPRESS)
    _add_gen_args(pb)
    pb.set_defaults(func=cmd_batch)

    psm = sub.add_parser("summary", help="summarize a batch output (counts, truncations, tokens, stamp)")
    psm.add_argument("file", type=Path)
    psm.set_defaults(func=cmd_summary)

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
