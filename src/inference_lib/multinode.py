"""``inference batch --nodes a,b,c``: one batch spread over several GPU nodes.

The coordinator (any host that can ssh to the nodes; paths must be on shared storage):

1. snapshots the remaining work once -- every (row, sample) not yet successful in the final
   output or any shard file -- and writes an explicit per-node assignment (round-robin), so
   workers never race over a moving work list;
2. starts one ordinary ``inference batch`` worker per node over ssh.  Each worker serves the
   model locally (with its own watchdog), processes its assignment and appends to
   ``<output>.shards/<node>.jsonl`` (resumable, fsync'd per row);
3. prints progress, then merges all shards into the output, checks every node ran the same
   stack (fingerprints in the shard stamps) and writes the combined stamp + summary.

Rerunning the same command resumes, also with a different node list.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .client import completed_keys, format_summary, merge_jsonl, summarize
from .registry import LIB_ROOT
from .stamp import fingerprint_diff, meta_path, write_stamp

SHARED_PREFIXES = ("/projects/",)


def shard_dir(output: Path) -> Path:
    return output.with_name(output.name + ".shards")


def plan_assignment(
    *, n_rows: int, n_samples: int, done: set[tuple[int, int]], nodes: list[str]
) -> dict[str, list[list[int]]]:
    todo = [(i, s) for i in range(n_rows) for s in range(n_samples) if (i, s) not in done]
    return {node: [list(k) for k in todo[j :: len(nodes)]] for j, node in enumerate(nodes)}


def _strip_nodes(argv: list[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--nodes":
            skip = True
            continue
        if a.startswith("--nodes="):
            continue
        out.append(a)
    return out


def _count_ok(path: Path) -> int:
    n = 0
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += not r.get("error") and r.get("responses") is not None
    return n


def run(args: Any, argv: list[str], *, n_rows: int, log=print) -> int:
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    output: Path = args.output.resolve()
    inp: Path = args.input.resolve()
    for p in (output, inp, LIB_ROOT, Path.cwd()):
        if not str(p).startswith(SHARED_PREFIXES):
            log(f"warning: {p} is not under {SHARED_PREFIXES}; the nodes may not see it")
    sdir = shard_dir(output)
    sdir.mkdir(parents=True, exist_ok=True)
    final_jsonl = output if output.suffix == ".jsonl" else output.with_name(output.name + ".partial.jsonl")
    sources = [final_jsonl, *sdir.glob("*.jsonl")]
    done = completed_keys([p for p in sources if p.exists()])
    token = uuid.uuid4().hex[:12]
    assignment = plan_assignment(n_rows=n_rows, n_samples=args.n, done=done, nodes=nodes)
    apath = sdir / f"assignment-{token}.json"
    apath.write_text(json.dumps(assignment))
    total = n_rows * args.n
    log(
        f"{total - len(done)}/{total} outputs to generate on {len(nodes)} nodes "
        f"({', '.join(f'{n}:{len(v)}' for n, v in assignment.items())}); shards in {sdir}"
    )

    worker_args = _strip_nodes(argv)
    procs: dict[str, subprocess.Popen] = {}
    for node in [n for n in nodes if assignment[n]]:  # nodes without work are not started
        remote = (
            f"cd {shlex.quote(os.getcwd())} && exec {shlex.quote(str(LIB_ROOT / '.venv/bin/inference'))} "
            + shlex.join(
                [
                    *worker_args,
                    "--input",
                    str(inp),
                    "--output",
                    str(output),
                    "--no-tqdm",
                    "--assignment",
                    str(apath),
                    "--shard-name",
                    node,
                    "--run-token",
                    token,
                ]
            )
            + f" > {shlex.quote(str(sdir / (node + '.log')))} 2>&1"
        )
        procs[node] = subprocess.Popen(["ssh", "-o", "BatchMode=yes", node, remote], stdin=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while any(p.poll() is None for p in procs.values()):
            time.sleep(60)
            parts = [f"{n}={_count_ok(sdir / (n + '.jsonl'))}" for n in nodes]
            ok = len(completed_keys([p for p in [final_jsonl, *sdir.glob("*.jsonl")] if p.exists()]))
            log(f"[{time.time() - t0:.0f}s] {ok}/{total} ok | " + " ".join(parts))
    except KeyboardInterrupt:
        log("interrupted: stopping workers (their servers shut down; rerun to resume)")
        for node in nodes:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", node, f"pkill -TERM -f 'run-token {token}'"], capture_output=True
            )
        for p in procs.values():
            p.wait(timeout=180)
        return 130
    codes = {n: p.returncode for n, p in procs.items()}
    for n, c in codes.items():
        if c not in (0, 2):
            log(f"worker {n} exited with {c}; see {sdir / (n + '.log')}")

    # merge + verify every node ran the same stack
    shard_files = sorted(sdir.glob("*.jsonl"))
    rows = merge_jsonl([p for p in [final_jsonl, *shard_files] if p.exists()], final_jsonl)
    stamps = {p.stem: json.loads(meta_path(p).read_text()) for p in shard_files if meta_path(p).exists()}
    fps = {n: s.get("fingerprint", {}) for n, s in stamps.items()}
    ref_node = next(iter(fps), None)
    mismatches = {n: fingerprint_diff(fps[ref_node], fp) for n, fp in fps.items() if ref_node and fp != fps[ref_node]}
    if mismatches:
        log(f"WARNING: nodes ran different stacks (vs {ref_node}): {json.dumps(mismatches, indent=2)}")
    summary = summarize(rows)
    summary["this_run"] = {"nodes": nodes, "wall_seconds": round(time.time() - t0, 1), "worker_exit_codes": codes}
    stamp = dict(stamps[ref_node]) if ref_node else {}
    stamp.pop("runs", None)
    stamp.update(
        multinode={"nodes": nodes, "shard_dir": str(sdir), "fingerprint_mismatches": mismatches},
        servers=[srv for s in stamps.values() for srv in s.get("servers", [])],
        runs=[r for s in stamps.values() for r in s.get("runs", [])],
        summary=summary,
    )
    write_stamp(output, stamp)
    if output != final_jsonl:
        from .io import detect_format, write_output

        write_output(output, rows, args.output_format or detect_format(output))
    log(format_summary(summary))
    log(f"wrote {output} (+ {meta_path(output).name})")
    complete = summary["ok"] == total and not mismatches
    return 0 if complete else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use `inference batch --nodes ...`")
