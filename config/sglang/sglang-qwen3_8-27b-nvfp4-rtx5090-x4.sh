#!/usr/bin/env bash
set -euo pipefail

# Each replica has a stable endpoint for assigning a complete document batch.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -eq 0 ]]; then
  set -- up
fi

exec docker compose \
  --env-file /dev/null \
  --project-name "${COMPOSE_PROJECT_NAME:-sglang-qwen3-8-27b-nvfp4-x4}" \
  --file "$script_dir/compose-qwen3_8-27b-nvfp4-rtx5090-x4.yaml" \
  "$@"
