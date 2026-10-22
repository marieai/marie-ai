#!/usr/bin/env bash

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MARIE_DEFAULT_MOUNT="${MARIE_DEFAULT_MOUNT:-${DATA_DIR:-$repo_dir}}"

PYTHONUNBUFFERED=1 MARIE_DEBUG=0 MARIE_DEBUG_PORT=5678 XXXXJINA_MP_START_METHOD=fork marie server --start --uses "$MARIE_DEFAULT_MOUNT/config/service/marie.yml"
