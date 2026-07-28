-- Summarize retained scheduler states and inspect outstanding work.
--
-- Optional psql variables:
--   queue_name       Restrict results to one queue name. Default: all queues.
--   lookback_minutes Recent-arrival window, clamped to 1-10080. Default: 60.
--   result_limit     Maximum detailed rows, clamped to 1-1000. Default: 200.
--
-- Example:
--   psql -v queue_name=corr -v lookback_minutes=60 -v result_limit=250 \
--     -f config/psql/schema/monitoring/outstanding_work_analysis.sql

\if :{?queue_name}
\else
\set queue_name ''
\endif

\if :{?result_limit}
\else
\set result_limit 200
\endif

\if :{?lookback_minutes}
\else
\set lookback_minutes 60
\endif

\pset pager off

-- Current state totals by queue. Outstanding work consists of created, retry,
-- and active jobs; terminal states remain visible for operational context.
WITH params AS (
    SELECT NULLIF(:'queue_name', '') AS queue_name
)
SELECT
    j.name AS queue_name,
    COUNT(*) AS retained_jobs,
    COUNT(*) FILTER (
        WHERE j.state::text IN ('created', 'retry', 'active')
    ) AS outstanding_jobs,
    COUNT(DISTINCT j.dag_id) FILTER (
        WHERE j.state::text IN ('created', 'retry', 'active')
    ) AS outstanding_dags,
    COUNT(*) FILTER (WHERE j.state::text = 'created') AS created_jobs,
    COUNT(*) FILTER (WHERE j.state::text = 'retry') AS retry_jobs,
    COUNT(*) FILTER (WHERE j.state::text = 'active') AS active_jobs,
    COUNT(*) FILTER (WHERE j.state::text = 'completed') AS completed_jobs,
    COUNT(*) FILTER (WHERE j.state::text = 'failed') AS failed_jobs,
    COUNT(*) FILTER (WHERE j.state::text = 'cancelled') AS cancelled_jobs,
    COUNT(*) FILTER (WHERE j.state::text = 'expired') AS expired_jobs,
    COUNT(*) FILTER (WHERE j.state::text = 'skipped') AS skipped_jobs,
    COUNT(*) FILTER (
        WHERE j.state::text = 'active'
          AND j.run_lease_expires_at <= NOW()
    ) AS active_with_expired_run_lease,
    MIN(j.created_on) FILTER (
        WHERE j.state::text IN ('created', 'retry', 'active')
    ) AS oldest_outstanding_created_on,
    MAX(NOW() - j.created_on) FILTER (
        WHERE j.state::text IN ('created', 'retry', 'active')
    ) AS oldest_outstanding_age
FROM marie_scheduler.job j
CROSS JOIN params p
WHERE p.queue_name IS NULL OR j.name = p.queue_name
GROUP BY j.name
ORDER BY outstanding_jobs DESC, j.name;

-- Recent arrivals. State counts describe the current state of jobs created in
-- the lookback window. The first row always reports the selected scope, even
-- when zero jobs arrived, followed by per-queue rows when data exists.
WITH params AS (
    SELECT
        NULLIF(:'queue_name', '') AS queue_name,
        LEAST(
            GREATEST(:'lookback_minutes'::integer, 1),
            10080
        ) AS lookback_minutes
), recent AS MATERIALIZED (
    SELECT j.*
    FROM marie_scheduler.job j
    CROSS JOIN params p
    WHERE j.created_on >= NOW() - make_interval(mins => p.lookback_minutes)
      AND (p.queue_name IS NULL OR j.name = p.queue_name)
), activity AS (
    SELECT
        0 AS display_order,
        COALESCE(MAX(p.queue_name), '[all queues]') AS queue_scope,
        COUNT(r.id) AS jobs_created,
        COUNT(r.id) FILTER (
            WHERE r.state::text IN ('created', 'retry', 'active')
        ) AS currently_outstanding,
        COUNT(r.id) FILTER (WHERE r.state::text = 'active') AS currently_active,
        COUNT(r.id) FILTER (WHERE r.state::text = 'completed') AS completed,
        COUNT(r.id) FILTER (WHERE r.state::text = 'failed') AS failed,
        MAX(r.created_on) AS last_job_created_on,
        NOW() - MAX(r.created_on) AS time_since_last_creation
    FROM params p
    LEFT JOIN recent r ON TRUE

    UNION ALL

    SELECT
        1,
        r.name,
        COUNT(*),
        COUNT(*) FILTER (
            WHERE r.state::text IN ('created', 'retry', 'active')
        ),
        COUNT(*) FILTER (WHERE r.state::text = 'active'),
        COUNT(*) FILTER (WHERE r.state::text = 'completed'),
        COUNT(*) FILTER (WHERE r.state::text = 'failed'),
        MAX(r.created_on),
        NOW() - MAX(r.created_on)
    FROM recent r
    CROSS JOIN params p
    WHERE p.queue_name IS NULL
    GROUP BY r.name
)
SELECT
    queue_scope,
    jobs_created,
    currently_outstanding,
    currently_active,
    completed,
    failed,
    last_job_created_on,
    time_since_last_creation
FROM activity
ORDER BY display_order, queue_scope;

-- Most recently created jobs in the lookback window. This reveals work that
-- arrived but completed or failed too quickly to remain in the capacity view.
WITH params AS (
    SELECT
        NULLIF(:'queue_name', '') AS queue_name,
        LEAST(
            GREATEST(:'lookback_minutes'::integer, 1),
            10080
        ) AS lookback_minutes,
        LEAST(GREATEST(:'result_limit'::integer, 1), 1000) AS result_limit
), recent AS MATERIALIZED (
    SELECT
        j.id,
        j.dag_id,
        j.name,
        j.state::text AS state,
        j.retry_count,
        j.retry_limit,
        j.created_on,
        j.started_on,
        j.completed_on,
        j.run_owner,
        j.run_attempt_id,
        j.output
    FROM marie_scheduler.job j
    CROSS JOIN params p
    WHERE j.created_on >= NOW() - make_interval(mins => p.lookback_minutes)
      AND (p.queue_name IS NULL OR j.name = p.queue_name)
    ORDER BY j.created_on DESC
    LIMIT (SELECT result_limit FROM params)
)
SELECT
    r.id AS job_id,
    r.dag_id,
    d.submission_name,
    d.planner,
    d.project_id,
    d.ref_type,
    d.ref_id,
    d.priority,
    d.task_count,
    r.name AS queue_name,
    r.state,
    r.retry_count,
    r.retry_limit,
    r.created_on,
    r.started_on,
    r.completed_on,
    CASE
        WHEN r.completed_on IS NOT NULL THEN r.completed_on - r.created_on
    END AS creation_to_terminal,
    r.run_owner,
    r.run_attempt_id,
    r.output->>'failure_source' AS failure_source,
    COALESCE(
        r.output->>'error_message',
        r.output->'error'->>'message'
    ) AS scheduler_error_message,
    kv.value->>'status' AS worker_status,
    kv.value->>'message' AS worker_message,
    env #>> '{attributes,host}' AS executor_host,
    env #>> '{attributes,runtime_name}' AS runtime_name,
    env #>> '{error,type}' AS executor_error_type,
    env #>> '{error,message}' AS executor_error_message
FROM recent r
LEFT JOIN marie_scheduler.dag d
    ON d.id = r.dag_id
LEFT JOIN marie_scheduler.kv_store_worker kv
    ON kv.namespace = 'job'
   AND kv.key = 'marie_internal/job_info_' || r.id::text
   AND kv.is_deleted = FALSE
LEFT JOIN LATERAL (
    SELECT COALESCE(
        NULLIF(kv.value->>'runtime_env_json', '')::jsonb,
        '{}'::jsonb
    ) AS env
) parsed ON TRUE
ORDER BY r.created_on DESC;

-- Oldest/highest-priority outstanding jobs, bounded before dependency and KV
-- lookups so this remains suitable for production diagnostics.
WITH params AS (
    SELECT
        NULLIF(:'queue_name', '') AS queue_name,
        LEAST(GREATEST(:'result_limit'::integer, 1), 1000) AS result_limit
), outstanding AS MATERIALIZED (
    SELECT
        j.id,
        j.dag_id,
        j.name,
        j.state::text AS state,
        j.priority,
        j.job_level,
        j.retry_count,
        j.retry_limit,
        j.created_on,
        j.start_after,
        j.started_on,
        j.lease_owner,
        j.lease_expires_at,
        j.run_owner,
        j.run_attempt_id,
        j.run_lease_expires_at
    FROM marie_scheduler.job j
    CROSS JOIN params p
    WHERE j.state::text IN ('created', 'retry', 'active')
      AND (p.queue_name IS NULL OR j.name = p.queue_name)
    ORDER BY
        CASE j.state::text
            WHEN 'active' THEN 0
            WHEN 'retry' THEN 1
            ELSE 2
        END,
        j.priority DESC,
        j.created_on
    LIMIT (SELECT result_limit FROM params)
)
SELECT
    o.id AS job_id,
    o.dag_id,
    d.submission_name,
    d.planner,
    d.project_id,
    d.ref_type,
    d.ref_id,
    d.priority,
    d.task_count,
    o.name AS queue_name,
    o.state,
    CASE
        WHEN o.state = 'active'
         AND kv.value->>'status' IN ('SUCCEEDED', 'FAILED', 'STOPPED')
        THEN 'terminal_kv_state_mismatch'
        WHEN o.state = 'active' AND o.run_owner IS NULL
        THEN 'missing_run_owner'
        WHEN o.state = 'active' AND o.run_lease_expires_at IS NULL
        THEN 'missing_run_lease'
        WHEN o.state = 'active' AND o.run_lease_expires_at <= NOW()
        THEN 'expired_run_lease'
        WHEN o.state = 'active'
        THEN 'running'
        WHEN blockers.unmet_dependencies > 0
        THEN 'waiting_dependencies'
        WHEN o.start_after > NOW()
        THEN 'delayed_until_start_after'
        ELSE 'ready_or_waiting_capacity'
    END AS outstanding_reason,
    o.job_level,
    o.retry_count,
    o.retry_limit,
    blockers.unmet_dependencies,
    NOW() - o.created_on AS outstanding_age,
    CASE
        WHEN o.started_on IS NOT NULL THEN NOW() - o.started_on
    END AS active_for,
    o.start_after,
    o.started_on,
    o.run_owner,
    o.run_attempt_id,
    o.run_lease_expires_at,
    kv.value->>'status' AS worker_status,
    kv.value->>'message' AS worker_message,
    env #>> '{attributes,host}' AS executor_host,
    env #>> '{attributes,runtime_name}' AS runtime_name,
    env #>> '{error,type}' AS last_error_type,
    env #>> '{error,message}' AS last_error_message
FROM outstanding o
LEFT JOIN marie_scheduler.dag d
    ON d.id = o.dag_id
LEFT JOIN marie_scheduler.kv_store_worker kv
    ON kv.namespace = 'job'
   AND kv.key = 'marie_internal/job_info_' || o.id::text
   AND kv.is_deleted = FALSE
LEFT JOIN LATERAL (
    SELECT COUNT(*)::integer AS unmet_dependencies
    FROM marie_scheduler.job_dependencies jd
    JOIN marie_scheduler.job dependency
      ON dependency.name = jd.depends_on_name
     AND dependency.id = jd.depends_on_id
    WHERE jd.job_name = o.name
      AND jd.job_id = o.id
      AND dependency.state::text NOT IN ('completed', 'skipped')
) blockers ON TRUE
LEFT JOIN LATERAL (
    SELECT COALESCE(
        NULLIF(kv.value->>'runtime_env_json', '')::jsonb,
        '{}'::jsonb
    ) AS env
) parsed ON TRUE
ORDER BY
    CASE o.state
        WHEN 'active' THEN 0
        WHEN 'retry' THEN 1
        ELSE 2
    END,
    o.priority DESC,
    o.created_on;
