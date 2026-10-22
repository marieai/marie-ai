#!/bin/sh
set -eu
clickhouse-client --host "${CLICKHOUSE_HOST:-localhost}" --port "${CLICKHOUSE_PORT:-9000}" \
  --user "${CLICKHOUSE_USER:-marie}" --multiquery <<'SQL'
ALTER USER IF EXISTS studio_query HOST NONE;
KILL QUERY WHERE user = 'studio_query' SYNC;
REVOKE ALL ON *.* FROM studio_query;
SQL
