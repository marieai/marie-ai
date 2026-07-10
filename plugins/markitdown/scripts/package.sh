#!/usr/bin/env bash
# Package the marie/markitdown plugin as a plain zip the daemon can decode.
# Usage: package.sh [OUTPUT_DIR]   (default: /mnt/data/marie-ai/plugins)
set -euo pipefail

plugin_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${1:-/mnt/data/marie-ai/plugins}"

version="$(
  python3 - "$plugin_dir/marie-extension.yaml" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
match = re.search(r'^\s*version:\s*"?([^"\n]+)"?', text, re.MULTILINE)
print(match.group(1).strip() if match else "0.0.0")
PY
)"

mkdir -p "$out_dir"
archive="$out_dir/marie-markitdown_${version}.zip"

python3 - "$plugin_dir" "$archive" <<'PY'
import sys, zipfile
plugin_dir, archive = sys.argv[1], sys.argv[2]
members = ["marie-extension.yaml", "main.py", "requirements.txt", "README.md"]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in members:
        zf.write(f"{plugin_dir}/{name}", arcname=name)
PY

echo "wrote $archive"
