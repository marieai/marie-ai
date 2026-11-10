#!/usr/bin/env bash
set -euo pipefail

# Single RTX 5090 (32 GB); see README.md for setup and memory tuning.
api_key_args=()
if [[ -n "${SGLANG_API_KEY:-}" ]]; then
  api_key_args=(--api-key "$SGLANG_API_KEY")
fi

BIND_IP=0.0.0.0

exec docker run --rm \
  --name "${CONTAINER_NAME:-sglang-qwen3-8-27b-nvfp4}" \
  --gpus "device=${GPU_ID:-0}" \
  --ipc=host \
  --publish "${BIND_IP:-127.0.0.1}:${PORT:-8000}:8000" \
  --volume "${HF_CACHE_DIR:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  --env HF_TOKEN \
  --env OMP_NUM_THREADS=1 \
  "${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.19}" \
  python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH:-nvidia/Qwen3.8-27B-NVFP4}" \
    --served-model-name Qwen3.8-27B \
    --host 0.0.0.0 \
    --port 8000 \
    --tp-size 1 \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-dtype float32 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-mamba-cache-size 4 \
    --max-running-requests 1 \
    --cuda-graph-max-bs-decode 1 \
    --context-length 32768 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --chunked-prefill-size 2048 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-metrics \
    "${api_key_args[@]}" \
    "$@"
