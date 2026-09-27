# inference-lib

Standalone vLLM serving and batch generation for 8xH100 nodes. Every model runs in a pinned
Docker image (below), and the `inference` CLI is a thin client (no torch or vLLM on the host).

```bash
git clone https://github.com/psidharth567/inference-library && cd inference-library
bash scripts/setup_env.sh                      # client venv at .venv (uv), seconds
bash scripts/pull_image.sh bodhanai-node001    # docker pull the image on a node (or run on the node)
export PATH=$PWD/.venv/bin:$PATH

# on a GPU node
inference list
inference serve qwen3-8b --detach              # one server, DP=8 on all 8 GPUs, :8000
inference chat -m qwen3-8b -p "Explain LoRA" --show-reasoning
inference batch -m qwen3-8b -i prompts.jsonl -o out.jsonl --temperature 0.6 --max-tokens 8192
inference stop
```

`chat` and `batch` start the model's deployment themselves when nothing is listening on `--port`
and stop it afterwards (`--keep-server` to leave it up, `--no-auto-serve` to require a running one).

## Images

| image | contents | used by |
|---|---|---|
| `ghcr.io/psidharth567/inference-library:vllm-v0.30.0` (public) | official `vllm/vllm-openai:v0.30.0`, identical layers + source label (`docker/vllm-v0.30.0`) | every preset |

## Models

Every preset (`configs/models/<key>.yaml`) uses the layout that measured fastest per GPU on one
8xH100 node, 1024 input / 1024 output tokens, 256 concurrent requests per GPU,
`vllm bench serve --ignore-eos`, BF16 KV cache. Measurements are from 2026-09-27.

| key | layout on 8 GPUs | out tok/s per GPU | measured as |
|---|---|---:|---|
| `qwen3-8b` | 1 server, TP1 x DP8 | 6,616 | whole node: 52.9k |
| `qwen3-14b` | 1 server, TP2 x DP4 | 4,351 | one TP2 replica (TP1: 4,262) |
| `qwen3-32b` | 1 server, TP2 x DP4 | 1,952 | one TP2 replica (TP4: 1,845) |
| `qwen3.5-9b` | 1 server, TP1 x DP8 | 8,756 | one GPU (TP2: 7,965) |
| `qwen3.5-35b-a3b` | 4 servers, TP2 each | 6,248 | whole node: 50.0k |
| `gemma4-26b-a4b-it` | 4 servers, TP2 each | 4,597 | whole node: 36.8k |
| `gemma4-31b-it` | 1 server, TP4 x DP2 | 1,294 | whole node: 10.4k |
| `olmo3-32b-think-dpo` | 1 server, TP2 x DP4 | 1,747 | one TP2 replica (TP4: 1,699) |
| `deepseek-r1-distill-llama-8b` | 1 server, TP1 x DP8 | 7,422 | one GPU |
| `deepseek-v4-flash` | 1 server, DP8 attention + EP8 | 2,877 | whole node: 23.0k at 4096 in flight |
| `glm-5.3-flash` | 1 server, TP2 x DP4 attention + EP8 | 1,072 | whole node: 8.6k at 1024 in flight |

"One replica" rows are per-GPU numbers from one server of the preset's size. Dense models'
in-server DP measured linear (Qwen3-8B: 6,679 tok/s on 1 GPU, 6,616 per GPU on 8), so those
numbers carry over to the whole node.

Any other HF repo id or local path works with vLLM defaults: `inference serve org/model --num-gpus 2`.

### Why these layouts

- **Dense models: one server, vLLM data parallel.** `--num-gpus 8` with TP=1 gives 8 DP ranks
  behind one endpoint, and they scale linearly (Qwen3-8B: 6,679 tok/s on 1 GPU, 52,931 on 8).
- **Small MoE models: independent servers.** vLLM's in-server DP runs MoE layers in lockstep
  across DP ranks. Qwen3.5-35B-A3B as TP2 x DP4 in one server gave 3,237 tok/s per GPU; as one
  DP8 + EP8 server 5,203; as 4 separate TP2 servers 6,248 (Gemma 4 26B-A4B: 3,232 / 4,462 /
  4,597). Presets with `data_parallel_size` split the GPUs into separate servers on consecutive
  ports (`--port`, `--port`+1, ...). `inference batch` finds them and round-robins requests.
- **Node-sized MoE models with MLA (DSV4-Flash, GLM-5.3-Flash): DP attention + EP.** They only
  fit on the whole node, and with TP the MLA latent KV is replicated on every rank. DSV4 with
  DP8+EP8 gave 23.0k tok/s vs 9.2k at TP8+EP (2.5x). GLM needs TP2 x DP4 (8.6k vs 6.3k at TP8):
  pure DP8 leaves too little memory for its KDA state (3.4k).
- **Attention / kernels.** vLLM 0.30 auto-selection was best or tied everywhere measured:
  FA3 for dense Qwen/Llama/OLMo (FlashInfer 1.5% slower), FA4 for Gemma 4 (Triton within 2%;
  FlashInfer rejects Gemma 4's bidirectional multimodal attention), FlashInfer GDN prefill for
  Qwen3.5 (FLA Triton and CuTe-DSL within 1%; decode uses FLA's recurrent kernels).
  `--performance-mode throughput` changed nothing (+0.3%) and OOMs Gemma 4.
- **Concurrency.** `inference batch` keeps `batch_concurrency` requests in flight (default 256
  per GPU, 4096 for DSV4). The server runs at most `max_num_seqs` per rank and queues the rest.

### Opt-in: FP8 KV cache

`--vllm-arg=--kv-cache-dtype=fp8` gives more concurrent sequences on KV-bound models: Qwen3-14B
+48% (4,262 -> 6,291), Qwen3-8B +29% (6,679 -> 8,594 at 512 concurrent). It is not a default.
On the full GSM8K test set (Qwen3-8B, thinking, greedy) accuracy went 95.38% -> 94.69%
(20 vs 11 discordant questions, McNemar p=0.15), and generations hit the 12k-token limit
3x as often (4 -> 12). Check it on your own task before using it.

## Layout and resolution

- Weights: explicit path > `$INFERENCE_MODELS_DIR/<key>` > `models/<key>` (a gitignored symlink
  farm, e.g. `models/qwen3-8b -> /projects/.../Qwen3-8B`) > `$HF_HOME` cache > download by repo
  id into `$HF_HOME` (default `.cache/huggingface`).
- Local model dirs are bind-mounted read-only at the same path (HF snapshots: the whole
  `models--org--name` dir, since snapshot files link into `../../blobs`).
- Logs: `logs/<container>.log`. Container names: `inference-<key>-<port>`.
- Per-server compile/JIT caches: `/tmp/inference-lib-<uid>/<container>` on the node
  (`$INFERENCE_CACHE_DIR`). The first start of a model on a node compiles kernels; later starts reuse them.

## Host constraints handled by the launcher

The nodes run driver 535 (CUDA 12.2). The notes below explain launcher choices that look odd.

- **CUDA 13 forward compatibility.** The image ships CUDA 13 user space, and
  `/usr/local/cuda/compat` goes first on `LD_LIBRARY_PATH`. The `v0.30.0-cu129` image tag is
  not usable here: it still contains a CUDA 13 torch and fails with "driver too old".
- **No `--ipc=host`.** Containers run as your uid, so no root-owned files land on /projects.
  With host IPC, systemd-logind's RemoveIPC deletes that uid's semaphores in /dev/shm whenever
  one of its ssh sessions closes, which kills multi-GPU startup (`SemLock ... FileNotFoundError`).
  Containers get a private `--shm-size 64g` instead.
- **Containers get an `/etc/passwd` entry for your uid** (the image's file plus one line). Older
  vLLM/torch call `getpwuid()` and crash without one.
- **Caches are node-local and per server.** When servers shared one cache on /projects,
  concurrent starts produced corrupt Inductor entries ("CUDA driver error: file not found")
  and missing FlashInfer cubins (`Assertion failed: !cubin.empty()`).
- **GLM-5.3-Flash / DSV4-Flash set `VLLM_ALLREDUCE_USE_FLASHINFER=0`.** With FlashInfer's
  one-shot all-reduce (mnnvl or trtllm), TP=8 startup hangs in CUDA-graph capture: rank 0
  blocks in a lazy kernel load while its GPU spins in the all-reduce. `CUDA_MODULE_LOADING=EAGER`
  also avoids the hang, but it stalls CUDA init for 10+ minutes.
- **GLM-5.3-Flash uses `--moe-backend deep_gemm`.** The auto-picked FLASHINFER_CUTLASS FP8 MoE
  crashed at the first MoE call (missing cubin). That happened while caches were still shared,
  so it may have been the cache race above. DeepGEMM is the configuration that was measured.
- **DeepSeek-R1-Distill-Llama-8B** declares `LlamaTokenizerFast` but ships a byte-level BPE
  `tokenizer.json`. transformers 5 rebuilds it as SentencePiece, so both encode and decode
  are wrong. The preset's `tokenizer_class: PreTrainedTokenizerFast` serves a patched copy of
  the tokenizer config. No other preset is affected (checked encode/decode round trips for all).

## CLI

```
inference list
inference serve MODEL [--port 8000] [--gpus 0,1 | --num-gpus N] [--max-model-len N] [-tp N]
                      [--vllm-arg=--flag=value ...] [--image IMG] [--no-docker] [--detach] [--dry-run]
inference stop [--port P]            # containers serving port P and above (all if omitted)
inference status [--port P]
inference chat  -m MODEL -p "..." [--system-prompt ...] [--thinking on|off] [--show-reasoning] [-o out.json]
inference batch -m MODEL -i IN -o OUT [--concurrency 256] [--temperature ..] [--top-p ..] [--top-k ..]
                [--seed ..] [--max-tokens ..] [--thinking on|off] [--extra-body JSON] [--dry-run]
inference bench [--port P] [--input-len 1024 --output-len 1024 --concurrency 256]
```

`--num-gpus` defaults to the preset's `recommended_gpus` (8). Free GPUs are picked automatically
unless you pass `--gpus`. `--no-docker` runs `vllm` from `PATH` / `$INFERENCE_VLLM_BIN` with
the same arguments.

### Batch I/O

Input is `jsonl`, `json`, `csv`, `tsv`, `txt`, `yaml` or `parquet` (by extension, or
`--input-format`). The prompt column is `prompt`/`prompts`/`text`/`input`/`question`/...
System prompt precedence: per-row `system` > `--system-prompt(-file)` / `$INFERENCE_SYSTEM_PROMPT` > none.

Output rows keep all input columns and add `_idx`, `responses`, `reasoning`, `finish_reason`,
`prompt_tokens`, `completion_tokens` and `error`. Every format streams through a JSONL file,
appended and fsync'd per row. Re-running the same command resumes: rows that already succeeded
are skipped and failed rows are retried. Non-JSONL outputs are written from that file at the end.

### Python

```python
from inference_lib import SamplingParams, get_client, run_batch, serve

with serve("qwen3.5-35b-a3b", num_gpus=8) as dep:       # 4 servers
    clients = [get_client(u) for u in dep.base_urls]
    rows = run_batch(client=clients, rows=[{"prompt": "hi"}], model=dep.served_model_name,
                     params=SamplingParams(max_tokens=512, temperature=0.7, enable_thinking=False))
```

## Tests

```bash
.venv/bin/python -m pytest -q tests      # no GPU needed
```
