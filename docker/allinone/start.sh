#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PROFILE="${1:-full}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/config/.env}"

# Ensure the external network exists
docker network create --driver=bridge marie_default 2>/dev/null || true

case "$PROFILE" in
  infra-only|infra)
    PROFILES=""
    ;;
  observability|obs)
    PROFILES="--profile observability"
    ;;
  application|app)
    PROFILES="--profile application"
    ;;
  full)
    PROFILES="--profile observability --profile application"
    ;;
  gpu)
    PROFILES="--profile observability --profile application --profile gpu"
    ;;
  *)
    echo "Usage: $0 {infra-only|observability|application|full|gpu}"
    exit 1
    ;;
esac

echo "Starting Marie-AI (profile: $PROFILE)..."
docker compose -f "$PROJECT_ROOT/Dockerfiles/docker-compose.allinone.yml" \
  --env-file "$ENV_FILE" \
  --project-directory "$PROJECT_ROOT" \
  $PROFILES up -d

echo "Marie-AI started. Use 'docker compose ps' to check status."
