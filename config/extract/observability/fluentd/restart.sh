#!/bin/sh
set -eu

LOG_DIR="/tmp/marie-healer"
mkdir -p "$LOG_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

TARGET_ID="${TARGET_CONTAINER_ID:-}"
TARGET_NAME="${TARGET_CONTAINER:-}"

if [ -z "$TARGET_ID" ] && [ -z "$TARGET_NAME" ]; then
  echo "[restart.sh] $(ts) ERROR: TARGET_CONTAINER_ID or TARGET_CONTAINER must be set" >&2
  exit 1
fi

# Cooldown / rate limit
COOLDOWN_SEC="${COOLDOWN_SEC:-300}"
WINDOW_SEC="${WINDOW_SEC:-600}"
MAX_PER_WINDOW="${MAX_PER_WINDOW:-3}"

key="${TARGET_NAME:-$TARGET_ID}"
state_last="$LOG_DIR/${key}.last_restart"
state_hist="$LOG_DIR/${key}.restart_hist"
log_file="$LOG_DIR/restarts.log"
meta_file="$LOG_DIR/last_restart.json"

now_epoch="$(date +%s)"
last_epoch="$( [ -f "$state_last" ] && cat "$state_last" || echo 0 )"
since_last=$((now_epoch - last_epoch))

# sliding window count
count=0
if [ -f "$state_hist" ]; then
  tmp="$LOG_DIR/.hist.$$"
  awk -v cutoff=$((now_epoch - WINDOW_SEC)) '{ if ($1 >= cutoff) { print; c++ } } END { if (c>0) {} }' "$state_hist" > "$tmp" || true
  mv "$tmp" "$state_hist"
  count="$(wc -l < "$state_hist" | tr -d ' ')"
fi

if [ "$since_last" -lt "$COOLDOWN_SEC" ]; then
  echo "[restart.sh] $(ts) Cooldown active (${since_last}s < ${COOLDOWN_SEC}s) — skipping." >&2
  exit 0
fi

if [ "$count" -ge "$MAX_PER_WINDOW" ]; then
  echo "[restart.sh] $(ts) Max restarts (${MAX_PER_WINDOW}) in ${WINDOW_SEC}s reached — skipping." >&2
  exit 0
fi

# Choose identifier & build REST call (no docker CLI required)
base="http://localhost"
sock="--unix-socket /var/run/docker.sock"
path="/containers/${TARGET_ID:-$TARGET_NAME}/restart"
echo "[restart.sh] $(ts) Restarting container '${TARGET_NAME:-$TARGET_ID}' ..." >&2

# 10s timeout is reasonable; adjust if needed
rc=0
out="$(curl -s -o /dev/null -w '%{http_code}' $sock -X POST "${base}${path}?t=10")" || rc=$?
if [ "$rc" -ne 0 ] || [ "$out" -ge 400 ]; then
  echo "[restart.sh] $(ts) ERROR: docker API restart failed (rc=$rc http=$out) for '${TARGET_NAME:-$TARGET_ID}'." >&2
  exit 1
fi

echo "$now_epoch" > "$state_last"
echo "$now_epoch" >> "$state_hist"
{
  printf '{'
  printf '"time":"%s",' "$(date -Iseconds)"
  printf '"target":"%s",' "${TARGET_NAME:-$TARGET_ID}"
  printf '"cooldown_sec":%s,' "$COOLDOWN_SEC"
  printf '"window_sec":%s,' "$WINDOW_SEC"
  printf '"max_per_window":%s' "$MAX_PER_WINDOW"
  printf '}\n'
} > "$meta_file"

echo "[restart.sh] $(ts) Restart complete." >&2
echo "$(ts) restarted ${TARGET_NAME:-$TARGET_ID}" >> "$log_file"

