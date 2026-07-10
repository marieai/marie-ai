#!/usr/bin/env bash
set -euo pipefail

# Single RTX 4090 / 24GB
# Qwen/Qwen3.5-9B with vLLM FP8 dynamic quantization

export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1

docker rm -f qwen35-9b-fp8 2>/dev/null || true

docker run --gpus all --ipc=host --restart unless-stopped \
  --shm-size 16g \
  --name qwen35-9b-fp8 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8008:8008 \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -e TORCH_HOME=/root/.cache/torch \
  -e CUDA_CACHE_PATH=/root/.nv/ComputeCache \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/torch:/root/.cache/torch" \
  -v "$HOME/.triton:/root/.triton" \
  -v "$HOME/.nv/ComputeCache:/root/.nv/ComputeCache" \
  -v "$HOME/.cache/models:/models" \
  -v "$HOME/.cache/vllm/torch_compile:/root/.cache/vllm/torch_compile_cache" \
  vllm/vllm-openai:qwen3_5 \
  --model Qwen/Qwen3.5-9B \
  --served-model-name qwen3.5-9b-fp8 \
  --download-dir /models \
  --tensor-parallel-size 1 \
  --quantization fp8 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --block-size 16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --port 8008
