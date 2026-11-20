#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${POSTGRES_PASSWORD:?Export POSTGRES_PASSWORD before starting fresh storage}"

exec docker compose --env-file /dev/null --project-directory "$repo_root" \
  -f "$repo_root/Dockerfiles/docker-compose.storage.yml" up --build -d psql
