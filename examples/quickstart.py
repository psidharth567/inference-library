"""Quickstart: input formats, system prompts, and a batch against a served model.

Run on a GPU node:  .venv/bin/python examples/quickstart.py [--serve]
"""

import sys
from pathlib import Path

from inference_lib import SamplingParams, get_client, get_prompt, get_system, load_input, run_batch, serve

EX = Path(__file__).parent

# 1. every input format loads to the same row dicts
for name in ["prompts.jsonl", "prompts.csv", "prompts.txt", "prompts.yaml"]:
    rows = load_input(EX / name)
    print(f"{name}: {len(rows)} rows, first prompt={get_prompt(rows[0])[:40]!r}")

# 2. system prompt precedence: per-row > global > none
for r in load_input(EX / "prompts_with_system.jsonl"):
    print(f"prompt={get_prompt(r)[:20]!r} system={get_system(r, 'You are helpful.')!r}")

# 3. serve + batch (needs GPUs and the vLLM image)
if "--serve" in sys.argv:
    with serve("qwen3-8b", num_gpus=1, port=8100) as dep:
        out = run_batch(
            client=[get_client(u) for u in dep.base_urls],
            rows=load_input(EX / "prompts.jsonl"),
            model=dep.served_model_name,
            params=SamplingParams(max_tokens=256, temperature=0.7, enable_thinking=False),
            use_tqdm=False,
        )
        for r in out:
            print(repr(r["responses"][:80]))
