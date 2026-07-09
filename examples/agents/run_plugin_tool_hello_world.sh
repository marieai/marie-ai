#!/usr/bin/env bash
#
# Live hello-world: call the installed `jina` plugin tool through the marie plugin
# daemon (marie-ai -> signed envelope -> /v1/dispatch/invoke -> plugin -> SSE).
#
# The daemon verifies an HMAC-signed envelope, so you MUST supply the same signing
# key the daemon (and Studio) were started with. Provide it either way:
#
#   1) export them, then run:
#        export MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID=...
#        export MARIE_PLUGIN_DAEMON_SIGNING_SECRET=...
#        ./examples/agents/run_plugin_tool_hello_world.sh
#
#   2) copy the template, fill in the signing key, and run:
#        cp examples/agents/.env.example examples/agents/.env
#        # edit examples/agents/.env -> set the two MARIE_PLUGIN_DAEMON_SIGNING_* values
#        ./examples/agents/run_plugin_tool_hello_world.sh
#      (or point at another file: MARIE_PLUGIN_ENV_FILE=/path/to/daemon.env ./...sh)
#
# Everything else defaults to the locally-installed jina plugin; override via env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # examples/agents -> marie-ai

# Optional: source a local .env holding the signing key (keep it out of git).
ENV_FILE="${MARIE_PLUGIN_ENV_FILE:-$SCRIPT_DIR/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

# --- Daemon + installed-plugin identity (override via env) -------------------
export MARIE_PLUGIN_DAEMON_URL="${MARIE_PLUGIN_DAEMON_URL:-http://127.0.0.1:8099}"
export MARIE_ORG_ID="${MARIE_ORG_ID:-7909fd70-2651-4c3d-86c8-c313652f63dd}"
export MARIE_WORKSPACE_ID="${MARIE_WORKSPACE_ID:-6fd6dd0f-171a-46ac-9b73-b6d6b9542981}"
export MARIE_PLUGIN_PACKAGE_REF="${MARIE_PLUGIN_PACKAGE_REF:-ext.langgenius.jina_tool}"
export MARIE_PLUGIN_PACKAGE_DIGEST="${MARIE_PLUGIN_PACKAGE_DIGEST:-sha256:70c638774b69dae2ab74717c3779de9de7ad8f6c768c227f2b5fce6db4635136}"
export MARIE_PLUGIN_PROVIDER_REF="${MARIE_PLUGIN_PROVIDER_REF:-jina}"
export MARIE_PLUGIN_TOOL_REF="${MARIE_PLUGIN_TOOL_REF:-jina_reader}"

# --- Signing key (REQUIRED; never hardcoded here) ----------------------------
if [ -z "${MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID:-}" ] || [ -z "${MARIE_PLUGIN_DAEMON_SIGNING_SECRET:-}" ]; then
  echo "ERROR: MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID / MARIE_PLUGIN_DAEMON_SIGNING_SECRET are not set." >&2
  echo "       Export them, or put them in: $ENV_FILE" >&2
  echo "       (the same signing key the daemon + Studio were started with)." >&2
  exit 1
fi

# Prefer the project venv if present; else fall back to python3 (e.g. conda env).
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

cd "$REPO_ROOT"
echo "daemon=$MARIE_PLUGIN_DAEMON_URL tool=$MARIE_PLUGIN_PROVIDER_REF/$MARIE_PLUGIN_TOOL_REF python=$PYTHON"
exec "$PYTHON" ./examples/agents/agent_plugin_tool_hello_world.py
