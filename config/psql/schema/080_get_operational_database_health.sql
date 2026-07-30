-- File: 080_get_operational_database_health.sql
-- Description: Safe PostgreSQL dependency-health aggregate

CREATE OR REPLACE FUNCTION marie_scheduler.get_operational_database_health()
RETURNS TABLE (
    active_sessions BIGINT,
    blocked_sessions BIGINT,
    oldest_transaction_seconds DOUBLE PRECISION
)
LANGUAGE SQL
STABLE
AS $function$
SELECT
    COUNT(*) FILTER (WHERE state = 'active')::BIGINT,
    COUNT(*) FILTER (WHERE cardinality(pg_blocking_pids(pid)) > 0)::BIGINT,
    MAX(EXTRACT(EPOCH FROM (NOW() - xact_start))) FILTER (
        WHERE xact_start IS NOT NULL
    )::DOUBLE PRECISION
FROM pg_stat_activity
WHERE datname = current_database();
$function$;

COMMENT ON FUNCTION marie_scheduler.get_operational_database_health()
IS 'Returns aggregate PostgreSQL activity without query text or connection details.';
