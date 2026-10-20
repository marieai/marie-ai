-- The container entrypoint creates the configured system identity (marie by default).
-- Its existing CLICKHOUSE_PASSWORD and access-management configuration own authentication.
-- Do not create another user or embed credentials in initialization SQL.
CREATE DATABASE IF NOT EXISTS telemetry;
