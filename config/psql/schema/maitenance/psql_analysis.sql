SELECT version();

SHOW shared_preload_libraries;
SHOW compute_query_id;
SHOW track_io_timing;
SHOW pg_stat_statements.track;
SHOW pg_stat_statements.max;

SELECT extname, extversion
FROM pg_extension
WHERE extname = 'pg_stat_statements';

SELECT *
FROM pg_stat_statements_info;


--- To reset all collected statistics:
SELECT pg_stat_statements_reset();

--- Check the collection window:

SELECT
    stats_reset,
    clock_timestamp() - stats_reset AS collection_duration,
    dealloc
FROM pg_stat_statements_info;

--- 3. Find queries with the greatest total impact

WITH statements AS (
    SELECT
        d.datname,
        r.rolname,
        p.*
    FROM pg_stat_statements AS p
    JOIN pg_database AS d
      ON d.oid = p.dbid
    JOIN pg_roles AS r
      ON r.oid = p.userid
    WHERE p.calls > 0
      AND p.query NOT ILIKE '%pg_stat_statements%'
)
SELECT
    datname,
    rolname,
    queryid,
    calls,

    round((total_exec_time / 1000)::numeric, 2)
        AS total_exec_seconds,

    round(mean_exec_time::numeric, 2)
        AS mean_exec_ms,

    round(max_exec_time::numeric, 2)
        AS max_exec_ms,

    round(stddev_exec_time::numeric, 2)
        AS stddev_exec_ms,

    round(
        (
            100.0 * total_exec_time
            / NULLIF(sum(total_exec_time) OVER (), 0)
        )::numeric,
        2
    ) AS percent_of_exec_time,

    round(
        (rows::numeric / NULLIF(calls, 0)),
        2
    ) AS rows_per_call,

    left(
        regexp_replace(query, E'[\\n\\r\\t ]+', ' ', 'g'),
        500
    ) AS query
FROM statements
ORDER BY total_exec_time DESC
LIMIT 30;


4. Find individually slow queries

SELECT
    d.datname,
    p.queryid,
    p.calls,

    round(p.mean_exec_time::numeric, 2)
        AS mean_ms,

    round(p.max_exec_time::numeric, 2)
        AS max_ms,

    round(p.stddev_exec_time::numeric, 2)
        AS stddev_ms,

    round(
        (
            p.stddev_exec_time
            / NULLIF(p.mean_exec_time, 0)
        )::numeric,
        2
    ) AS variability_ratio,

    round((p.total_exec_time / 1000)::numeric, 2)
        AS total_seconds,

    left(
        regexp_replace(p.query, E'[\\n\\r\\t ]+', ' ', 'g'),
        500
    ) AS query
FROM pg_stat_statements AS p
JOIN pg_database AS d
  ON d.oid = p.dbid
WHERE p.calls >= 5
  AND p.query NOT ILIKE '%pg_stat_statements%'
ORDER BY p.mean_exec_time DESC
LIMIT 30;


10. Examine the actual plan

Once you identify a problematic queryid, retrieve its normalized SQL:

SELECT
    queryid,
    calls,
    mean_exec_time,
    max_exec_time,
    query
FROM pg_stat_statements
WHERE queryid = -259577614151508021;

The stored query is normalized, so literal values are replaced with parameters such as $1. Queries with the same structure are aggregated together.

EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    FORMAT TEXT
)
    ???????????????????/

5. Find physical I/O-heavy queries


SELECT
    d.datname,
    p.queryid,
    p.calls,

    pg_size_pretty(
        (
            p.shared_blks_read
            * current_setting('block_size')::bigint
        )::bigint
    ) AS data_read,

    pg_size_pretty(
        (
            (p.shared_blks_hit + p.shared_blks_read)
            * current_setting('block_size')::bigint
        )::bigint
    ) AS data_accessed,

    round(
        (
            100.0 * p.shared_blks_hit
            / NULLIF(
                p.shared_blks_hit + p.shared_blks_read,
                0
            )
        )::numeric,
        2
    ) AS shared_hit_percent,

    round((p.shared_blk_read_time / 1000)::numeric, 2)
        AS read_time_seconds,

    round(
        (
            p.shared_blks_read::numeric
            / NULLIF(p.calls, 0)
        ),
        2
    ) AS blocks_read_per_call,

    round(p.mean_exec_time::numeric, 2)
        AS mean_ms,

    left(
        regexp_replace(p.query, E'[\\n\\r\\t ]+', ' ', 'g'),
        500
    ) AS query
FROM pg_stat_statements AS p
JOIN pg_database AS d
  ON d.oid = p.dbid
WHERE p.shared_blks_read > 0
  AND p.query NOT ILIKE '%pg_stat_statements%'
ORDER BY p.shared_blk_read_time DESC,
         p.shared_blks_read DESC
LIMIT 30;


6. Find temporary-file spills

SELECT
    d.datname,
    p.queryid,
    p.calls,

    pg_size_pretty(
        (
            p.temp_blks_written
            * current_setting('block_size')::bigint
        )::bigint
    ) AS temp_written,

    pg_size_pretty(
        (
            p.temp_blks_read
            * current_setting('block_size')::bigint
        )::bigint
    ) AS temp_read,

    round((p.temp_blk_write_time / 1000)::numeric, 2)
        AS temp_write_seconds,

    round((p.temp_blk_read_time / 1000)::numeric, 2)
        AS temp_read_seconds,

    round(p.mean_exec_time::numeric, 2)
        AS mean_ms,

    left(
        regexp_replace(p.query, E'[\\n\\r\\t ]+', ' ', 'g'),
        500
    ) AS query
FROM pg_stat_statements AS p
JOIN pg_database AS d
  ON d.oid = p.dbid
WHERE p.temp_blks_written > 0
   OR p.temp_blks_read > 0
ORDER BY
    p.temp_blks_written + p.temp_blks_read DESC
LIMIT 30;


7. Find excessively frequent queries

SELECT
    d.datname,
    p.queryid,
    p.calls,

    round(
        (
            p.calls
            / NULLIF(
                extract(
                    epoch FROM
                    clock_timestamp() - p.stats_since
                ),
                0
            )
        )::numeric,
        2
    ) AS calls_per_second,

    round(p.mean_exec_time::numeric, 3)
        AS mean_ms,

    round((p.total_exec_time / 1000)::numeric, 2)
        AS total_seconds,

    round(
        (
            p.rows::numeric
            / NULLIF(p.calls, 0)
        ),
        2
    ) AS rows_per_call,

    left(
        regexp_replace(p.query, E'[\\n\\r\\t ]+', ' ', 'g'),
        500
    ) AS query
FROM pg_stat_statements AS p
JOIN pg_database AS d
  ON d.oid = p.dbid
WHERE p.calls > 0
  AND p.query NOT ILIKE '%pg_stat_statements%'
ORDER BY calls_per_second DESC
LIMIT 30;


1. Pull the complete statistics for this query
SELECT
    queryid,
    calls,

    round(total_plan_time::numeric, 2) AS total_plan_ms,
    round(mean_plan_time::numeric, 3) AS mean_plan_ms,

    round(total_exec_time::numeric, 2) AS total_exec_ms,
    round(mean_exec_time::numeric, 2) AS mean_exec_ms,
    round(min_exec_time::numeric, 2) AS min_exec_ms,
    round(max_exec_time::numeric, 2) AS max_exec_ms,
    round(stddev_exec_time::numeric, 2) AS stddev_exec_ms,

    rows,
    round(rows::numeric / NULLIF(calls, 0), 2) AS rows_per_call,

    shared_blks_hit,
    shared_blks_read,
    shared_blks_dirtied,
    shared_blks_written,

    round(shared_blk_read_time::numeric, 2) AS read_time_ms,
    round(shared_blk_write_time::numeric, 2) AS write_time_ms,

    temp_blks_read,
    temp_blks_written,
    round(temp_blk_read_time::numeric, 2) AS temp_read_time_ms,
    round(temp_blk_write_time::numeric, 2) AS temp_write_time_ms,

    wal_records,
    wal_fpi,
    pg_size_pretty(wal_bytes::bigint) AS wal_generated,

    query
FROM pg_stat_statements
WHERE queryid = -259577614151508021;

How to interpret it
High shared_blks_read and shared_blk_read_time: likely large scans or missing indexes.
High shared_blks_hit but low physical reads: the query is touching too many cached pages.
High temp_blks_written: sort, hash, or aggregate operations are spilling to disk.
Large max_exec_time or standard deviation: intermittent blocking, changing parameters, or unstable plans.
Low I/O but high execution time: CPU-heavy JSON processing, function loops, or expensive expressions.
High planning time: complex dynamically generated statements or repeated replanning.
2. Calculate blocks touched per call



2. Calculate blocks touched per call

This is particularly useful for a scheduler query:

SELECT
    queryid,
    calls,

    round(
        shared_blks_hit::numeric / NULLIF(calls, 0),
        2
    ) AS cache_blocks_per_call,

    round(
        shared_blks_read::numeric / NULLIF(calls, 0),
        2
    ) AS disk_blocks_per_call,

    round(
        (
            shared_blks_hit + shared_blks_read
        )::numeric / NULLIF(calls, 0),
        2
    ) AS total_blocks_per_call,

    round(mean_exec_time::numeric, 2) AS mean_exec_ms,

    round(
        rows::numeric / NULLIF(calls, 0),
        2
    ) AS rows_per_call
FROM pg_stat_statements
WHERE queryid = -259577614151508021;



3. Inspect the function definition
SELECT
    p.oid::regprocedure AS function_signature,
    l.lanname AS language,
    p.provolatile,
    p.proparallel,
    p.prosecdef AS security_definer,
    pg_get_functiondef(p.oid) AS function_definition
FROM pg_proc AS p
JOIN pg_namespace AS n
  ON n.oid = p.pronamespace
JOIN pg_language AS l
  ON l.oid = p.prolang
WHERE n.nspname = 'marie_scheduler'
  AND p.proname = 'admission_candidate_dags';



EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT
    candidate.dag_id,
    candidate.serialized_dag
FROM marie_scheduler.admission_candidate_dags(
    100,
    600,
    ARRAY[]::uuid[]
) WITH ORDINALITY AS candidate(
    dag_id,
    serialized_dag,
    admission_rank
)
ORDER BY candidate.admission_rank;


EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT
    d.id,
    d.serialized_dag
FROM marie_scheduler.dag d
WHERE d.state IN ('created', 'active')
  AND NOT (
      d.id = ANY(COALESCE(ARRAY[]::uuid[], ARRAY[]::uuid[]))
  )
  AND EXISTS (
      SELECT 1
      FROM marie_scheduler.job ready
      WHERE ready.dag_id = d.id
        AND ready.state IN ('created', 'retry')
        AND ready.start_after <= CURRENT_TIMESTAMP
  )
  AND NOT EXISTS (
      SELECT 1
      FROM marie_scheduler.job blocker
      WHERE blocker.dag_id = d.id
        AND blocker.state IN ('failed', 'expired', 'cancelled')
  )
ORDER BY
    d.priority DESC,
    COALESCE(d.soft_sla, d.hard_sla) ASC NULLS LAST,
    d.created_on,
    d.id
LIMIT GREATEST(0, 100);

8. Find write and WAL-heavy queries

SELECT
    d.datname,
    p.queryid,
    p.calls,

    pg_size_pretty(p.wal_bytes::bigint)
        AS total_wal,

    pg_size_pretty(
        (
            p.wal_bytes
            / NULLIF(p.calls, 0)
        )::bigint
    ) AS wal_per_call,

    p.wal_records,
    p.wal_fpi,

    p.shared_blks_dirtied,
    p.shared_blks_written,

    round((p.total_exec_time / 1000)::numeric, 2)
        AS total_seconds,

    left(
        regexp_replace(p.query, E'[\\n\\r\\t ]+', ' ', 'g'),
        500
    ) AS query
FROM pg_stat_statements AS p
JOIN pg_database AS d
  ON d.oid = p.dbid
WHERE p.wal_bytes > 0
ORDER BY p.wal_bytes DESC
LIMIT 30;


SELECT
    a.pid,
    a.usename,
    a.datname,
    a.application_name,

    clock_timestamp() - a.query_start
        AS query_duration,

    a.wait_event_type,
    a.wait_event,

    pg_blocking_pids(a.pid)
        AS blocking_pids,

    a.query_id,

    left(
        regexp_replace(a.query, E'[\\n\\r\\t ]+', ' ', 'g'),
        500
    ) AS query
FROM pg_stat_activity AS a
WHERE a.state <> 'idle'
  AND a.pid <> pg_backend_pid()
ORDER BY a.query_start;


9. Look for active blocking separately

SELECT
    a.pid,
    a.usename,
    a.datname,
    a.application_name,

    clock_timestamp() - a.query_start
        AS query_duration,

    a.wait_event_type,
    a.wait_event,

    pg_blocking_pids(a.pid)
        AS blocking_pids,

    a.query_id,

    left(
        regexp_replace(a.query, E'[\\n\\r\\t ]+', ' ', 'g'),
        500
    ) AS query
FROM pg_stat_activity AS a
WHERE a.state <> 'idle'
  AND a.pid <> pg_backend_pid()
ORDER BY a.query_start;