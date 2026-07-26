#!/usr/bin/env bash

set -uo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/check-port-health.sh [PORT ...]

Inspect TCP port ownership, process state, CLOSE_WAIT owners, zombies, and
orphaned Marie/Python processes on the current host.

Examples:
  scripts/check-port-health.sh
  scripts/check-port-health.sh 52318 58131 62177
  scripts/check-port-health.sh 50011 63365
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

for command in ss lsof ps awk sort xargs find sed readlink hostname wc; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        exit 1
    fi
done

ports=("$@")
for port in "${ports[@]}"; do
    if [[ ! "$port" =~ ^[0-9]+$ ]] || ((10#$port < 1 || 10#$port > 65535)); then
        echo "Invalid TCP port: $port" >&2
        exit 2
    fi
done

sudo_command=()
if ((EUID != 0)); then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "This script requires root privileges or sudo." >&2
        exit 1
    fi
    sudo -v || exit 1
    sudo_command=(sudo)
fi

check_port() {
    local port="$1"
    local listener_rows
    local pid
    local stat
    local -a pids

    echo
    echo "===== $(hostname -f 2>/dev/null || hostname) port=$port ====="

    echo "--- LISTEN socket ---"
    listener_rows=$(
        "${sudo_command[@]}" ss -H -ltnp "sport = :$port" 2>/dev/null || true
    )
    if [[ -n "$listener_rows" ]]; then
        printf '%s\n' "$listener_rows"
    else
        echo "No LISTEN socket found."
    fi

    echo "--- All TCP states involving port ---"
    "${sudo_command[@]}" ss -H -tanp \
        "( sport = :$port or dport = :$port )" 2>/dev/null || true

    echo "--- lsof ownership ---"
    "${sudo_command[@]}" lsof -nP -iTCP:"$port" 2>/dev/null || true

    mapfile -t pids < <(
        "${sudo_command[@]}" lsof -nP -t -iTCP:"$port" 2>/dev/null | sort -u
    )

    if ((${#pids[@]} == 0)); then
        if [[ -n "$listener_rows" ]]; then
            echo "RESULT: a listener exists, but lsof did not resolve its owner."
        else
            echo "RESULT: no process owns TCP port $port."
            echo "If etcd still registers it, the registration is stale."
        fi
        return
    fi

    for pid in "${pids[@]}"; do
        stat=$(
            "${sudo_command[@]}" ps -o stat= -p "$pid" 2>/dev/null | xargs
        )

        echo
        echo "--- PID $pid ---"
        "${sudo_command[@]}" ps \
            -o pid=,ppid=,user=,stat=,lstart=,etime=,cmd= -p "$pid" || true
        printf 'executable: '
        "${sudo_command[@]}" readlink -f "/proc/$pid/exe" || true
        printf 'open descriptors: '
        "${sudo_command[@]}" find "/proc/$pid/fd" \
            -maxdepth 1 -type l 2>/dev/null | wc -l
        "${sudo_command[@]}" sed -n '/Max open files/p' "/proc/$pid/limits" || true
        echo "cgroup:"
        "${sudo_command[@]}" sed -n '1,10p' "/proc/$pid/cgroup" || true

        case "$stat" in
            Z*) echo "RESULT: PID $pid is a zombie; it cannot own an active socket." ;;
            D*) echo "RESULT: PID $pid is stuck in uninterruptible sleep." ;;
            "") echo "RESULT: PID $pid exited while being inspected." ;;
            *) echo "RESULT: PID $pid is alive with state=$stat." ;;
        esac
    done
}

scan_processes() {
    local rows

    echo
    echo "===== Host-wide CLOSE_WAIT owners ====="
    rows=$(
        "${sudo_command[@]}" lsof -nP -iTCP -sTCP:CLOSE_WAIT 2>/dev/null |
            awk 'NR > 1 {
                owner=$1 " pid=" $2 " user=" $3
                count[owner]++
            }
            END {
                for (owner in count) print count[owner], owner
            }' |
            sort -nr
    )
    [[ -n "$rows" ]] && printf '%s\n' "$rows" || echo "No CLOSE_WAIT sockets found."

    echo
    echo "===== Zombie processes ====="
    rows=$(
        "${sudo_command[@]}" ps -eo pid=,ppid=,user=,stat=,etime=,cmd= |
            awk '$4 ~ /^Z/ {print}'
    )
    [[ -n "$rows" ]] && printf '%s\n' "$rows" || echo "No zombie processes found."

    echo
    echo "===== Orphaned Marie/Python processes (PPID 1) ====="
    rows=$(
        "${sudo_command[@]}" ps -eo pid=,ppid=,user=,stat=,etime=,cmd= |
            awk '$2 == 1 && tolower($0) ~ /(marie|python)/ {print}'
    )
    [[ -n "$rows" ]] && printf '%s\n' "$rows" || \
        echo "No orphaned Marie/Python processes found."
}

for port in "${ports[@]}"; do
    check_port "$port"
done

scan_processes
