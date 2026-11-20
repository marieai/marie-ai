#!/bin/bash
set -euo pipefail

[[ -f /tmp/marie-postgres-ready ]] || exit 1
exec /usr/local/bin/documentdb-healthcheck
