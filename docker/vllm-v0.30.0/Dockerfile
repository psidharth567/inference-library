# Default serving image: the official vLLM 0.30.0 release image, unchanged (same layers), plus
# the label that links the GHCR package to this repo. The launcher supplies everything
# host-specific at run time (CUDA compat path, user, caches), see src/inference_lib/server.py.
FROM vllm/vllm-openai:v0.30.0
LABEL org.opencontainers.image.source="https://github.com/psidharth567/inference-library" \
      org.opencontainers.image.description="vLLM 0.30.0 (official image, unchanged) for inference-library" \
      org.opencontainers.image.base.name="docker.io/vllm/vllm-openai:v0.30.0"
