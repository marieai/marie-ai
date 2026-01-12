#!/bin/sh
set -eu

LOG_DIR="/tmp/marie-healer"
mkdir -p "$LOG_DIR"
TRIG_LOG="$LOG_DIR/trigger.log"
REC_FILE="$(mktemp "$LOG_DIR/rec.XXXXXX.json")"

# capture matched record from stdin (Fluentd exec passes JSON)
cat > "$REC_FILE"

{
  echo "----- $(date -Iseconds) -----"
  echo "[trigger] env:"
  echo "  TARGET_CONTAINER_ID=${TARGET_CONTAINER_ID:-}"
  echo "  TARGET_CONTAINER=${TARGET_CONTAINER:-}"
  echo "  COOLDOWN_SEC=${COOLDOWN_SEC:-}"
  echo "  WINDOW_SEC=${WINDOW_SEC:-}"
  echo "  MAX_PER_WINDOW=${MAX_PER_WINDOW:-}"
  echo "[trigger] record (first 1k):"
  head -c 1000 "$REC_FILE" || true
  echo
  echo "[trigger] calling restart.sh..."
} >> "$TRIG_LOG"

if /etc/marie/config/extract/observability/fluentd/restart.sh >>"$TRIG_LOG" 2>&1; then
  echo "[trigger] restart.sh OK" >> "$TRIG_LOG"
else
  rc=$?
  echo "[trigger] restart.sh FAILED rc=$rc" >> "$TRIG_LOG"
  exit $rc
fi

