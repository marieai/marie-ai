#!/bin/sh
set -eu

directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Reuse the reader credential across schema updates. Operators can supply a mounted
# password file; local Compose generates it in the existing ClickHouse volume.
if [ -z "${QUERY_READER_PASSWORD_FILE:-}" ]; then
  QUERY_READER_PASSWORD_FILE=/var/lib/clickhouse/query-explorer/reader-password
  if [ ! -s "$QUERY_READER_PASSWORD_FILE" ]; then
    (
      umask 077
      mkdir -p "$(dirname -- "$QUERY_READER_PASSWORD_FILE")"
      head -c 64 /dev/urandom | sha256sum | cut -d ' ' -f 1 > "$QUERY_READER_PASSWORD_FILE"
    )
  fi
fi
if [ ! -s "$QUERY_READER_PASSWORD_FILE" ]; then
  echo 'The query reader password file is empty or unavailable' >&2
  exit 1
fi
reader_hash=$(tr -d '\r\n' < "$QUERY_READER_PASSWORD_FILE" | sha256sum | cut -d ' ' -f 1)
definer_hash=$(head -c 64 /dev/urandom | sha256sum | cut -d ' ' -f 1)

run() {
  clickhouse-client --host "${CLICKHOUSE_HOST:-localhost}" --port "${CLICKHOUSE_PORT:-9000}" \
    --user "${CLICKHOUSE_USER:-default}" --multiquery
}

run <<SQL
CREATE DATABASE IF NOT EXISTS query;
CREATE USER IF NOT EXISTS studio_query_definer IDENTIFIED WITH sha256_hash BY '$definer_hash' HOST NONE;
CREATE USER IF NOT EXISTS studio_query IDENTIFIED WITH sha256_hash BY '$reader_hash' DEFAULT DATABASE query;
ALTER USER studio_query IDENTIFIED WITH sha256_hash BY '$reader_hash' HOST ANY DEFAULT DATABASE query;
REVOKE ALL ON *.* FROM studio_query;
REVOKE ALL ON *.* FROM studio_query_definer;
GRANT SELECT ON otel.otel_traces TO studio_query_definer;
GRANT SELECT ON otel.otel_logs TO studio_query_definer;
GRANT SELECT ON otel.span_annotations TO studio_query_definer;
GRANT SELECT ON otel.model_prices TO studio_query_definer;
GRANT dictGet ON otel.model_price_dict TO studio_query_definer;
GRANT SELECT ON query.spans TO studio_query_definer;
CREATE SETTINGS PROFILE IF NOT EXISTS studio_query_limits;
ALTER SETTINGS PROFILE studio_query_limits SETTINGS
  readonly = 1 CONST,
  log_queries = 0 CONST,
  max_execution_time = 30 CONST,
  timeout_overflow_mode = 'throw' CONST,
  max_result_rows = 20000 CONST,
  max_result_bytes = 16777216 CONST,
  result_overflow_mode = 'throw' CONST,
  max_rows_to_read = 100000000 CONST,
  max_bytes_to_read = 10737418240 CONST,
  read_overflow_mode = 'throw' CONST,
  max_memory_usage = 1073741824 CONST,
  max_memory_usage_for_user = 2147483648 CONST,
  max_threads = 2 CONST,
  max_concurrent_queries_for_user = 4 CONST,
  max_query_size = 65536 CONST,
  output_format_json_quote_64bit_integers = 1 CONST,
  output_format_json_quote_decimals = 1 CONST,
  cancel_http_readonly_queries_on_client_close = 1 CONST,
  replace_running_query = 1 CONST,
  replace_running_query_max_wait_ms = 5000 CONST
TO studio_query;
CREATE QUOTA IF NOT EXISTS studio_query_hourly;
ALTER QUOTA studio_query_hourly FOR INTERVAL 1 HOUR MAX queries = 120, execution_time = 600 TO studio_query;
SQL
run < "$directory/views.sql"
run <<'SQL'
GRANT SELECT ON query.spans TO studio_query;
GRANT SELECT ON query.traces TO studio_query;
GRANT SELECT ON query.logs TO studio_query;
GRANT SELECT ON query.annotations TO studio_query;
GRANT SELECT ON query.model_prices TO studio_query;
GRANT SELECT ON query.schema_version TO studio_query;
GRANT SELECT(quota_name, duration, max_queries, max_execution_time) ON system.quota_usage TO studio_query;
SQL

printf 'SQL explorer schema ready. Reader password file: %s\n' "$QUERY_READER_PASSWORD_FILE"
