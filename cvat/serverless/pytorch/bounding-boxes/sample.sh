#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
    echo "Usage: $0 IMAGE_FILE" >&2
    exit 2
fi

image=$(base64 < "$1" | tr -d '\n')
cat << EOF > /tmp/input.json
{"image": "$image"}
EOF
cat /tmp/input.json | nuctl invoke pth.marieai.bboxes -c 'application/json'
