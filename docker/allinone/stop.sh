#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/config/.env}"

echo "Stopping Marie-AI..."
docker compose -f "$PROJECT_ROOT/Dockerfiles/docker-compose.allinone.yml" \
  --env-file "$ENV_FILE" \
  --project-directory "$PROJECT_ROOT" \
  --profile observability --profile application --profile gpu \
  down "$@"

echo "Marie-AI stopped."
