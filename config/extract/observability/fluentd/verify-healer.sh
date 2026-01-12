#!/bin/sh
# verify-healer.sh — end-to-end tests for Fluentd Log Healer (Docker API)
# Runs on the HOST. Requires: docker, curl. (jq optional)
set -eu

# ---------- Defaults (override via env or CLI --target/--healer/--state) ----------
TARGET_CONTAINER="${TARGET_CONTAINER:-marieai-dev-server}"
HEALER_NAME="${HEALER_NAME:-marieai-dev-log-healer}"
FLUENTD_MONITOR_URL="${FLUENTD_MONITOR_URL:-http://127.0.0.1:24220/api/plugins.json}"
STATE_DIR="${STATE_DIR:-/tmp/marie-healer}"

# ---------- Formatting ----------
ok()   { printf '\033[32m✔ %s\033[0m\n' "$*"; }
warn() { printf '\033[33m⚠ %s\033[0m\n' "$*"; }
err()  { printf '\033[31m✘ %s\033[0m\n' "$*"; }
info() { printf '∙ %s\n' "$*"; }

# ---------- Helpers ----------
die() { err "$*"; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

docker_exec() {
  # $1: container; rest: command
  docker exec -i "$1" /bin/sh -lc "$(
    shift
    printf '%s' "$*"
  )"
}

json_get() {
  # jq if available, else cat
  if have jq; then jq -r "$1"; else cat; fi
}

usage() {
cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --target NAME        Target app container (default: ${TARGET_CONTAINER})
  --healer NAME        Healer sidecar container (default: ${HEALER_NAME})
  --state PATH         Healer state dir on host (default: ${STATE_DIR})
  --reset              Clear cooldown/rate-limit state & markers
  --simulate           Append a synthetic CUDA OOM log to trigger auto-restart
  --force              Bypass cooldown for the manual restart test
  --no-manual          Skip the manual restart test
  --help               Show this help

Examples:
  $(basename "$0") --reset --force
  TARGET_CONTAINER=my-app HEALER_NAME=my-healer $(basename "$0") --simulate
EOF
}

# ---------- Parse CLI ----------
RESET=0 SIMULATE=0 FORCE=0 SKIP_MANUAL=0
while [ "${1:-}" ]; do
  case "$1" in
    --target)    TARGET_CONTAINER="$2"; shift 2;;
    --healer)    HEALER_NAME="$2"; shift 2;;
    --state)     STATE_DIR="$2"; shift 2;;
    --reset)     RESET=1; shift;;
    --simulate)  SIMULATE=1; shift;;
    --force)     FORCE=1; shift;;
    --no-manual) SKIP_MANUAL=1; shift;;
    --help|-h)   usage; exit 0;;
    *) err "Unknown option: $1"; usage; exit 1;;
  esac
done

# ---------- Pre-flight ----------
have docker || die "docker is required on the host"
have curl   || die "curl is required on the host"

info "Target container   : ${TARGET_CONTAINER}"
info "Healer container   : ${HEALER_NAME}"
info "Fluentd monitor URL: ${FLUENTD_MONITOR_URL}"
info "Healer state dir   : ${STATE_DIR}"

# ---------- 1) Check containers up ----------
info "Checking containers are running…"
docker ps --format '{{.Names}}' | grep -qx "${HEALER_NAME}" \
  && ok "Healer container is running (${HEALER_NAME})" \
  || die "Healer container not running: ${HEALER_NAME}"

docker ps --format '{{.Names}}' | grep -qx "${TARGET_CONTAINER}" \
  && ok "Target container is running (${TARGET_CONTAINER})" \
  || die "Target container not running: ${TARGET_CONTAINER}"

# ---------- 2) Fluentd health ----------
info "Checking Fluentd monitor endpoint…"
if curl -fsS "${FLUENTD_MONITOR_URL}" >/dev/null; then
  ok "Fluentd monitor endpoint is healthy"
else
  die "Fluentd monitor endpoint not responding: ${FLUENTD_MONITOR_URL}"
fi

# ---------- 3) Docker Engine socket from inside HEALER ----------
info "Validating Docker Engine API is reachable from inside healer…"
if docker_exec "${HEALER_NAME}" 'curl --unix-socket /var/run/docker.sock -fsS http://localhost/_ping' 2>/dev/null | grep -q '^OK'; then
  ok "Engine API reachable via /var/run/docker.sock"
else
  die "Engine API NOT reachable from healer (check /var/run/docker.sock mount & curl in container)"
fi

# ---------- 4) Optional: reset state ----------
if [ "$RESET" -eq 1 ]; then
  info "Resetting healer state under ${STATE_DIR}…"
  rm -f "${STATE_DIR}/${TARGET_CONTAINER}.last_restart" \
        "${STATE_DIR}/${TARGET_CONTAINER}.restart_hist" \
        "${STATE_DIR}"/restart-"${TARGET_CONTAINER}"-*.marker 2>/dev/null || true
  ok "State cleared (cooldown/rate-limit & markers)"
fi

# ---------- 5) Manual restart (uses restart.sh) ----------
if [ "$SKIP_MANUAL" -eq 0 ]; then
  info "Triggering MANUAL restart via restart.sh (inside healer)…"
  # If --force, override cooldown only; rate-limit still enforced by script
  if [ "$FORCE" -eq 1 ]; then
    CMD="TARGET_CONTAINER=${TARGET_CONTAINER} COOLDOWN_SEC=0 /etc/marie/config/extract/observability/fluentd/restart.sh"
  else
    CMD="TARGET_CONTAINER=${TARGET_CONTAINER} /etc/marie/config/extract/observability/fluentd/restart.sh"
  fi
  if docker_exec "${HEALER_NAME}" "${CMD}"; then
    ok "Manual restart script returned success"
  else
    warn "Manual restart reported an error (could be cooldown/rate-limit). Check ${STATE_DIR} & healer logs."
  fi
fi

# ---------- 6) Validate state artifacts ----------
info "Checking state artifacts under ${STATE_DIR}…"
[ -d "${STATE_DIR}" ] || die "State dir missing: ${STATE_DIR}"
[ -f "${STATE_DIR}/restarts.log" ] && ok "restarts.log present" || warn "restarts.log not present yet"
[ -f "${STATE_DIR}/last_restart.json" ] && ok "last_restart.json present" || warn "last_restart.json not present yet"

# Show last restart JSON (if jq is present)
if [ -f "${STATE_DIR}/last_restart.json" ]; then
  info "last_restart.json (pretty-printed if jq is available):"
  if have jq; then jq . "${STATE_DIR}/last_restart.json" || true
  else cat "${STATE_DIR}/last_restart.json"; echo; fi
fi

# ---------- 7) Optional: simulate a GPU OOM log ----------
if [ "$SIMULATE" -eq 1 ]; then
  info "Simulating a CUDA OOM log line in the target container’s JSON log…"
  # Get container ID and log path
  CID="$(docker inspect -f '{{.Id}}' "${TARGET_CONTAINER}")" || die "Cannot inspect ${TARGET_CONTAINER}"
  LOG_DIR="/var/lib/docker/containers/${CID}"
  LOG_FILE="${LOG_DIR}/${CID}-json.log"

  # Append a synthetic JSON log line (stderr stream)
  LINE='{"log": "RuntimeError: CUDA out of memory (synthetic test)", "stream": "stderr", "time": "'"$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"'"}'

  if docker_exec "${TARGET_CONTAINER}" "test -w ${LOG_FILE} || true" >/dev/null 2>&1; then
    # Write from inside the target container if possible
    docker_exec "${TARGET_CONTAINER}" "printf '%s\n' '$LINE' >> ${LOG_FILE}"
    ok "Injected synthetic OOM line from inside target container"
  else
    # Fallback: write from host (requires root or permissions on /var/lib/docker)
    if [ -w "${LOG_FILE}" ]; then
      printf '%s\n' "$LINE" | sudo tee -a "${LOG_FILE}" >/dev/null
      ok "Injected synthetic OOM line from host"
    else
      warn "Could not write to Docker JSON log (permissions). Skipping simulation."
    fi
  fi

  info "Give Fluentd a couple seconds to process…"
  sleep 3
  # Show tail of healer logs
  docker logs --tail=50 "${HEALER_NAME}" 2>/dev/null || true
fi

ok "All checks completed."

