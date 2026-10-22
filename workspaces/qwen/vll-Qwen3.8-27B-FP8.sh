#!/usr/bin/env bash
set -euo pipefail

# vLLM launch for Qwen3.8-27B (hybrid dense+sparse MoE)
#
# Model : Qwen/Qwen3.8-27B  (~27B params, hybrid Gated-DeltaNet + MoE)
# Native ctx : 262 144 tokens
# Features : thinking mode, vision, tool-calling, MTP speculative decoding

# ── runtime knobs ──────────────────────────────────────────────────


export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1

export NCCL_IB_DISABLE=0
# export NCCL_P2P_LEVEL=NVL

docker run \
  --gpus all \
  --ipc=host \
  --restart unless-stopped \
  --shm-size 12g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8888:8888 \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -e TORCH_HOME=/root/.cache/torch \
  -e CUDA_CACHE_PATH=/root/.nv/ComputeCache \
  -e VLLM_ENABLE_CUDAGRAPH_GC=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_P2P_LEVEL=NVL \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/torch:/root/.cache/torch" \
  -v "$HOME/.triton:/root/.triton" \
  -v "$HOME/.nv/ComputeCache:/root/.nv/ComputeCache" \
  -v "$HOME/.cache/models:/models" \
  -v "$HOME/.cache/vllm/torch_compile:/root/.cache/vllm/torch_compile_cache" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.8-27B-FP8 \
  --served-model-name qwen3-vl-instruct \
  --download-dir /models \
  --dtype auto \
  --tensor-parallel-size 1 \
  --block-size 32 \
  --max-model-len 46768 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 8888
