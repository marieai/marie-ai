#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: install-zoekt.sh /absolute/binary/directory' >&2
  exit 2
fi
revision=d64299b4420d0f0a46082c26d34ee956184558ed
mkdir -p "$1"
GOBIN="$1" GOTOOLCHAIN=auto go install "github.com/sourcegraph/zoekt/cmd/zoekt-index@$revision"
GOBIN="$1" GOTOOLCHAIN=auto go install "github.com/sourcegraph/zoekt/cmd/zoekt-webserver@$revision"
