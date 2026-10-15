#!/bin/sh
set -eu
if [ -n "${CLICKHOUSE_ADMIN_PASSWORD_FILE:-}" ]; then
  CLICKHOUSE_PASSWORD=$(cat "$CLICKHOUSE_ADMIN_PASSWORD_FILE")
  export CLICKHOUSE_PASSWORD
fi
clickhouse-client --host "${CLICKHOUSE_HOST:-localhost}" --port "${CLICKHOUSE_PORT:-9000}" \
  --user "${CLICKHOUSE_USER:-default}" --multiquery <<'SQL'
ALTER USER IF EXISTS studio_query HOST NONE;
KILL QUERY WHERE user = 'studio_query' SYNC;
REVOKE ALL ON *.* FROM studio_query;
SQL
