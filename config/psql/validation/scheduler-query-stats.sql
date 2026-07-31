SELECT
    queryid,
    calls,
    round(total_exec_time::numeric, 1) AS total_exec_ms,
    round(mean_exec_time::numeric, 3) AS mean_exec_ms,
    round(max_exec_time::numeric, 3) AS max_exec_ms,
    round(rows::numeric / NULLIF(calls, 0), 2) AS rows_per_call,
    shared_blks_hit,
    shared_blks_read,
    temp_blks_read,
    temp_blks_written,
    wal_records,
    wal_fpi,
    pg_size_pretty(wal_bytes::bigint) AS wal_size,
    left(regexp_replace(query, '\s+', ' ', 'g'), 300) AS query
FROM pg_stat_statements
WHERE query ~* 'hydrate_frontier_jobs|admission_candidate_dags|lease_jobs_by_id|activate_from_lease|release_expired_leases|claim_expired_run_leases|extend_run_lease|resolve_dag_state'
ORDER BY total_exec_time DESC;
