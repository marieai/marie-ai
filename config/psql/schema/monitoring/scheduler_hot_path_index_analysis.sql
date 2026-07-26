-- Run the EXPLAIN statements before and after applying
-- 073_scheduler_hot_path_indexes.sql. They do not update scheduler rows.

SELECT jobid, jobname, schedule, command, active
FROM cron.job
WHERE jobname IN (
    'refresh_job_priority',
    'refresh_job_durations',
    'refresh_dag_durations'
)
ORDER BY jobname;

SELECT
    table_name,
    index_name,
    index_definition
FROM (
    SELECT
        tree.relid::regclass::text AS table_name,
        index_class.relname AS index_name,
        pg_get_indexdef(index_row.indexrelid) AS index_definition
    FROM pg_partition_tree('marie_scheduler.job'::regclass) AS tree
    JOIN pg_index AS index_row ON index_row.indrelid = tree.relid
    JOIN pg_class AS index_class ON index_class.oid = index_row.indexrelid
) AS indexes
WHERE index_definition ILIKE '%lease_expires_at%'
   OR index_definition ILIKE '%run_lease_expires_at%'
ORDER BY table_name, index_name;

EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT j.id, j.name
FROM marie_scheduler.job AS j
WHERE j.state IN ('created', 'retry')
  AND j.lease_owner IS NOT NULL
  AND j.lease_expires_at IS NOT NULL
  AND j.lease_expires_at <= now()
ORDER BY j.lease_expires_at, j.id
LIMIT 1000;

EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT j.id, j.name
FROM marie_scheduler.job AS j
WHERE j.state = 'active'
  AND j.run_owner IS NOT NULL
  AND j.run_attempt_id IS NOT NULL
  AND j.run_lease_expires_at IS NOT NULL
  AND j.run_lease_expires_at <= now()
ORDER BY j.run_lease_expires_at, j.id
LIMIT 1000;

SELECT
    calls,
    round(total_exec_time::numeric, 1) AS total_ms,
    round(mean_exec_time::numeric, 3) AS mean_ms,
    round(max_exec_time::numeric, 3) AS max_ms,
    shared_blks_hit,
    shared_blks_read,
    left(regexp_replace(query, '\s+', ' ', 'g'), 240) AS query
FROM pg_stat_statements
WHERE query ILIKE '%release_expired_leases%'
   OR query ILIKE '%claim_expired_run_leases%'
ORDER BY total_exec_time DESC;
