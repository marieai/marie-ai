#!/usr/bin/env bash
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
archive="$out_dir/marie-plugin-document-extraction_${version}.zip"

python3 - "$plugin_dir" "$archive" <<'PY'
from pathlib import Path
import sys
import zipfile

plugin_dir = Path(sys.argv[1])
archive = Path(sys.argv[2])
roots = [
    plugin_dir / "main.py",
    plugin_dir / "marie-extension.yaml",
    plugin_dir / "pyproject.toml",
    plugin_dir / "uv.lock",
    plugin_dir / "README.md",
]
roots.extend(sorted((plugin_dir / "marie_plugins").rglob("*.py")))
roots.extend(sorted((plugin_dir / "marie_plugins").rglob("*.scm")))
roots.extend(sorted((plugin_dir / "marie_plugins").rglob("ATTRIBUTION.md")))
roots.extend(sorted((plugin_dir / "schemas").glob("*.json")))

with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
    for path in roots:
        if path.is_file():
            output.write(path, arcname=path.relative_to(plugin_dir).as_posix())
PY

echo "wrote $archive"
