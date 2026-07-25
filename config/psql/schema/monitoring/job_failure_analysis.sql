-- Analyze one scheduler job across PostgreSQL state, worker KV history, and
-- durable execution attempts.
--
-- Replace the UUID below, then execute this entire file in DataGrip, DBeaver,
-- pgAdmin, or psql. The setting is transaction-local and leaves no database
-- objects or session configuration behind.

BEGIN;
SET LOCAL marie_monitor.job_id = '00000000-0000-0000-0000-000000000000';

-- Current scheduler and worker-KV state.
WITH target(job_id) AS (
    SELECT CAST(current_setting('marie_monitor.job_id') AS uuid)
)
SELECT
    j.id AS job_id,
    j.dag_id,
    j.name AS queue_name,
    j.data->'metadata'->>'ref_type' AS ref_type,
    j.data->'metadata'->>'ref_id' AS ref_id,
    j.state::text AS scheduler_state,
    j.retry_count,
    j.retry_limit,
    j.run_attempt_id,
    j.started_on,
    j.completed_on,
    j.output->>'failure_source' AS failure_source,
    j.output->>'error_message' AS scheduler_error_message,
    j.output->'error' AS scheduler_error,
    kv.value->>'status' AS worker_status,
    kv.value->>'message' AS worker_message,
    to_timestamp(
        NULLIF(kv.value->>'end_time', '')::bigint / 1000.0
    ) AS worker_end_time
FROM target t
LEFT JOIN marie_scheduler.job j
    ON j.id = t.job_id
LEFT JOIN marie_scheduler.kv_store_worker kv
    ON kv.namespace = 'job'
   AND kv.key = 'marie_internal/job_info_' || t.job_id::text
   AND kv.is_deleted = FALSE;

-- Structured executor exceptions. JobInfo.message can be the generic
-- "Job failed." while the underlying exception is stored in runtime_env_json.
WITH target(job_id) AS (
    SELECT CAST(current_setting('marie_monitor.job_id') AS uuid)
)
SELECT
    kh.change_time,
    kh.value->>'run_attempt_id' AS run_attempt_id,
    env #>> '{attributes,host}' AS executor_host,
    env #>> '{attributes,executor}' AS executor,
    env #>> '{attributes,runtime_name}' AS runtime_name,
    env #>> '{attributes,executor_endpoint}' AS endpoint,
    env #>> '{error,type}' AS error_type,
    env #>> '{error,message}' AS error_message,
    env #>> '{error,filename}' AS error_file,
    env #>> '{error,name}' AS error_function,
    env #>> '{error,line_no}' AS error_line,
    env #>> '{error,traceback}' AS full_traceback
FROM target t
JOIN marie_scheduler.kv_store_worker_history kh
    ON kh.namespace = 'job'
   AND kh.key = 'marie_internal/job_info_' || t.job_id::text
CROSS JOIN LATERAL (
    SELECT COALESCE(
        NULLIF(kh.value->>'runtime_env_json', '')::jsonb,
        '{}'::jsonb
    ) AS env
) parsed
WHERE kh.value->>'status' = 'FAILED'
ORDER BY kh.change_time;

-- Combined state and attempt timeline. This distinguishes the original worker
-- failure from retries and later terminal-event replays.
WITH target(job_id) AS (
    SELECT CAST(current_setting('marie_monitor.job_id') AS uuid)
), events AS (
    SELECT
        jh.history_created_on AS event_time,
        'job_history'::text AS source,
        jh.state::text AS status,
        COALESCE(
            jh.output->>'error_message',
            jh.output->'error'->>'message'
        ) AS message,
        jsonb_strip_nulls(jsonb_build_object(
            'run_attempt_id', jh.run_attempt_id,
            'retry_count', jh.retry_count,
            'failure_source', jh.output->>'failure_source',
            'error', jh.output->'error'
        )) AS details
    FROM marie_scheduler.job_history jh
    JOIN target t ON t.job_id = jh.id

    UNION ALL

    SELECT
        kh.change_time,
        'kv_history',
        kh.value->>'status',
        kh.value->>'message',
        jsonb_strip_nulls(jsonb_build_object(
            'operation', kh.operation,
            'error_type', env #>> '{error,type}',
            'error_message', env #>> '{error,message}',
            'executor_host', env #>> '{attributes,host}',
            'runtime_name', env #>> '{attributes,runtime_name}',
            'run_owner', kh.value->>'run_owner',
            'run_attempt_id', kh.value->>'run_attempt_id'
        ))
    FROM target t
    JOIN marie_scheduler.kv_store_worker_history kh
        ON kh.namespace = 'job'
       AND kh.key = 'marie_internal/job_info_' || t.job_id::text
    CROSS JOIN LATERAL (
        SELECT COALESCE(
            NULLIF(kh.value->>'runtime_env_json', '')::jsonb,
            '{}'::jsonb
        ) AS env
    ) parsed

    UNION ALL

    SELECT
        COALESCE(
            ja.terminal_at,
            ja.dispatch_confirmed_at,
            ja.dispatch_started_at,
            ja.activated_at
        ),
        'job_attempt',
        ja.attempt_state,
        COALESCE(ja.dispatch_error, ja.terminal_reject_reason),
        jsonb_strip_nulls(jsonb_build_object(
            'run_attempt_id', ja.run_attempt_id,
            'terminal_status', ja.terminal_status,
            'terminal_work_state', ja.terminal_work_state,
            'terminal_source', ja.terminal_source,
            'terminal_accepted', ja.terminal_accepted,
            'gateway_instance_id', ja.gateway_instance_id,
            'run_owner', ja.run_owner
        ))
    FROM marie_scheduler.job_attempt ja
    JOIN target t ON t.job_id = ja.job_id
)
SELECT event_time, source, status, message, details
FROM events
ORDER BY event_time, source;

COMMIT;
