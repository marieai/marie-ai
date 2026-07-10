#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="bartowski/Qwen_Qwen3.5-9B-GGUF"
MODEL_FILE="Qwen_Qwen3.5-9B-Q8_0.gguf"
MODEL_DIR="$HOME/.cache/models/qwen35-9b-gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

CONTAINER_NAME="qwen35-9b-gguf"
PORT=8000

echo "==> Preparing directories"
mkdir -p "$MODEL_DIR"

echo "==> Checking for model"
if [ ! -f "$MODEL_PATH" ]; then
  echo "==> Downloading GGUF model from Hugging Face"

  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" \
      --include "$MODEL_FILE" \
      --local-dir "$MODEL_DIR"
  else
    echo "huggingface-cli not found, using curl fallback"

    # NOTE: public repo, no token needed
    URL="https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE"
    curl -L "$URL" -o "$MODEL_PATH"
  fi
else
  echo "==> Model already exists, skipping download"
fi

echo "==> Pulling vLLM image"
docker pull vllm/vllm-openai:qwen3_5

echo "==> Stopping existing container (if any)"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "==> Starting vLLM server"

docker run --gpus all --ipc=host --restart unless-stopped \
  --shm-size 16g \
  --name "$CONTAINER_NAME" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "$PORT:8000" \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -e TORCH_HOME=/root/.cache/torch \
  -e CUDA_CACHE_PATH=/root/.nv/ComputeCache \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/torch:/root/.cache/torch" \
  -v "$HOME/.nv/ComputeCache:/root/.nv/ComputeCache" \
  -v "$MODEL_DIR:/models" \
  vllm/vllm-openai:qwen3_5 \
  --model /models/$MODEL_FILE \
  --tokenizer Qwen/Qwen3.5-9B \
  --served-model-name qwen3.5-9b-q8-gguf \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --block-size 16 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --port 8000

echo "==> Waiting for server to be ready..."
sleep 5

echo "==> Available models:"
curl -s http://localhost:$PORT/v1/models | jq || true

echo ""
echo "==> Test request:"
curl http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b-q8-gguf",
    "messages": [
      {"role": "user", "content": "Say hello from GGUF on a 4090."}
    ],
    "temperature": 0.2
  }'