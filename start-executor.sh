#!/usr/bin/env bash

set -e

export PYTHONUNBUFFERED=1
export COLUMNS=180
export CUDA_VISIBLE_DEVICES=0
export GRPC_VERBOSITY=debug
export JINA_MP_START_METHOD=spawn
export MARIE_CACHE_LOCK_TIMEOUT=10
export MARIE_DEBUG=0
export MARIE_DEBUG_PORT=5678
export MARIE_DEFAULT_MOUNT=/mnt/data/marie-ai

echo "[INFO] Environment variables set. Starting application..."


python marie  server --start --uses /mnt/data/marie-ai/config/service/extract/marie-extract-4.0.0.yml

# python ~/dev/marieai/marie-ai/marie/__main__.py server --start --uses /mnt/data/marie-ai/config/service/extract/marie-extract-4.0.0.yml 
