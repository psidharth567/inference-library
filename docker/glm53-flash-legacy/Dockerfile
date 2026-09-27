# Legacy GLM-5.3-Flash stack, kept bit-for-bit. Same layers as the local image toolkit/inference-glm53:12.8
# (image id sha256:6a5c3471f919...), which is vllm/vllm-openai:glm53-flash (vLLM nightly
# 0.1.dev20051+g487ecf187, torch 2.13.0+cu130, transformers 5.15.1, FlashInfer 0.6.17) plus env:
#   LD_LIBRARY_PATH=/usr/local/cuda/compat:..., NCCL_NVLS_ENABLE=1, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64,
#   VLLM_MARLIN_USE_ATOMIC_ADD=1, VLLM_ALLREDUCE_USE_FLASHINFER=1, VLLM_USE_BREAKABLE_CUDAGRAPH=0
FROM toolkit/inference-glm53:12.8
LABEL org.opencontainers.image.source="https://github.com/psidharth567/inference-library" \
      org.opencontainers.image.description="Legacy GLM-5.3-Flash stack (vLLM 0.1.dev20051+g487ecf187)"
