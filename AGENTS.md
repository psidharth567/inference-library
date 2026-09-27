# inference-lib — agent instructions

Standalone vLLM serving + batch generation on 8xH100 nodes (driver 535). No dependency on any
other toolkit library. Read `README.md` for the measured presets and the host constraints the
launcher handles; this file covers how to use it.

```bash
export INFERENCE_LIB_ROOT=/projects/data/llmteam/sidharth/inference-library   # github.com/psidharth567/inference-library
export PATH=$INFERENCE_LIB_ROOT/.venv/bin:$PATH      # bash scripts/setup_env.sh if missing
```

GPU nodes: `ssh -o BatchMode=yes bodhanai-node0XX`. Docker works without sudo there. `/projects`
is shared, but each node has its own `/tmp`. Run `inference serve/chat/batch` ON the GPU node.
Pull the default image with `scripts/pull_image.sh <host>...`.

## Translate a request into this first

```yaml
inference_request:
  model: <registry key | HF repo | local path>   # `inference list`
  mode: serve | chat | batch | bench
  input: /abs/path/prompts.{jsonl,json,csv,tsv,txt,yaml,parquet}
  output: /abs/path/out.{jsonl,json,csv,tsv,txt,yaml}   # under the project, not in this library
  system_prompt: null | "..." | file
  sampling: {max_tokens: 2048, temperature: 0.7, top_p: null, top_k: null, seed: null, thinking: null|on|off}
  gpus: null (preset: 8, auto-picked free) | "0,1,2,3" | num_gpus: N
  port: 8000
```

## Recipes

```bash
# serve and leave running (MoE presets start several servers on port, port+1, ...)
inference serve qwen3.5-35b-a3b --detach
inference status; inference stop

# batch: auto-serves when nothing is on --port, stops afterwards; resumable (re-run same command)
inference batch -m glm-5.3-flash -i in.jsonl -o out.jsonl --max-tokens 16384 --temperature 1.0 --top-p 0.95
inference batch -m qwen3-8b -i in.csv -o out.csv --thinking off --system-prompt-file sys.txt

# against servers that are already running (several replicas: comma-separated)
inference batch -m qwen3.5-35b-a3b --base-url http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1 -i in.jsonl -o out.jsonl

# fewer GPUs / specific GPUs / extra vLLM flags / print the docker command only
inference serve qwen3-8b --gpus 0,1 --vllm-arg=--kv-cache-dtype=fp8 --dry-run

# throughput check against a running server
inference bench --port 8000 --concurrency 256
```

Output rows: the input columns plus `_idx, responses, reasoning, finish_reason, prompt_tokens,
completion_tokens, error`. Reasoning models return their thinking in `reasoning`, not
`responses`. Check `finish_reason == "length"` counts before trusting a run: that is
truncation at `--max-tokens`.

## Rules

- Serve through `inference serve` / `Deployment`, not a hand-written `docker run`. The launcher
  applies the fixes for this cluster (CUDA compat, private IPC, per-server node-local caches,
  preset env). If you need the raw command, use `--dry-run`.
- New model: add `configs/models/<key>.yaml` (fields = `ModelSpec` in `registry.py`; unknown
  fields are rejected) and a `models/<key>` symlink, or rely on HF download. Measure layouts
  with `inference bench` before claiming a preset is optimal. Record the numbers in a comment
  in the preset and in the README table.
- MoE models: set `data_parallel_size` (independent servers), unless a single-server
  layout is measured faster for that model.
- Keep BF16 KV as the default; FP8 KV is opt-in (README has the quality measurement).
- Big models (GLM-5.3-Flash, DSV4-Flash) take 5-15 min to start cold. Don't kill them before
  the preset's timeout; follow `logs/inference-<key>-<port>.log`.
- Don't write outputs, caches or logs of project runs into this library. `.cache/`, `logs/`,
  `models/` and `.work/` are gitignored.
- `ruff check src tests` and `pytest -q tests` before committing.
