-- File: 074_get_operational_dag.sql
-- Description: Payload-free operational DAG detail
-- Dependencies: 007_dag.sql, 008_dag_history.sql, 065_job_attempt.sql

CREATE OR REPLACE FUNCTION {schema}.get_operational_dag(
    p_dag_id UUID,
    p_queued_too_long_seconds INTEGER DEFAULT 300,
    p_running_too_long_seconds INTEGER DEFAULT 900,
    p_stale_update_seconds INTEGER DEFAULT 600
)
RETURNS TABLE (
    dag_id UUID,
    dag_name TEXT,
    dag_state TEXT,
    planner TEXT,
    priority INTEGER,
    task_count INTEGER,
    created_on TIMESTAMPTZ,
    started_on TIMESTAMPTZ,
    completed_on TIMESTAMPTZ,
    updated_on TIMESTAMPTZ,
    age_seconds DOUBLE PRECISION,
    last_update_age_seconds DOUBLE PRECISION,
    jobs_total BIGINT,
    jobs_created BIGINT,
    jobs_retry BIGINT,
    jobs_active BIGINT,
    jobs_completed BIGINT,
    jobs_skipped BIGINT,
    jobs_expired BIGINT,
    jobs_cancelled BIGINT,
    jobs_failed BIGINT,
    queues TEXT[],
    queued_too_long BIGINT,
    running_too_long BIGINT,
    stale_update BIGINT,
    retrying BIGINT,
    failed_attention BIGINT,
    terminal_mismatch BIGINT,
    lifecycle JSONB
)
LANGUAGE SQL
STABLE
ROWS 1
AS $function$
WITH job_stats AS (
    SELECT
        j.dag_id,
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE j.state::TEXT = 'created') AS created,
        COUNT(*) FILTER (WHERE j.state::TEXT = 'retry') AS retry,
        COUNT(*) FILTER (WHERE j.state::TEXT = 'active') AS active,
        COUNT(*) FILTER (WHERE j.state::TEXT = 'completed') AS completed,
        COUNT(*) FILTER (WHERE j.state::TEXT = 'skipped') AS skipped,
        COUNT(*) FILTER (WHERE j.state::TEXT = 'expired') AS expired,
        COUNT(*) FILTER (WHERE j.state::TEXT = 'cancelled') AS cancelled,
        COUNT(*) FILTER (WHERE j.state::TEXT = 'failed') AS failed,
        ARRAY_AGG(DISTINCT j.name ORDER BY j.name) AS queues,
        MAX(
            COALESCE(ja.updated_on, j.completed_on, j.started_on, j.created_on)
        ) AS last_updated_on,
        COUNT(*) FILTER (
            WHERE j.state::TEXT IN ('created', 'retry')
              AND EXTRACT(EPOCH FROM (
                    NOW() - COALESCE(
                        ja.updated_on,
                        j.completed_on,
                        j.started_on,
                        j.created_on
                    )
                  )) > p_queued_too_long_seconds
        ) AS queued_too_long,
        COUNT(*) FILTER (
            WHERE j.state::TEXT = 'active'
              AND j.started_on IS NOT NULL
              AND EXTRACT(EPOCH FROM (NOW() - j.started_on))
                    > p_running_too_long_seconds
        ) AS running_too_long,
        COUNT(*) FILTER (
            WHERE j.state::TEXT IN ('active', 'retry')
              AND EXTRACT(EPOCH FROM (
                    NOW() - COALESCE(
                        ja.updated_on,
                        j.completed_on,
                        j.started_on,
                        j.created_on
                    )
                  )) > p_stale_update_seconds
        ) AS stale_update,
        COUNT(*) FILTER (WHERE j.state::TEXT = 'retry') AS retrying,
        COUNT(*) FILTER (
            WHERE j.state::TEXT IN ('failed', 'expired', 'cancelled')
        ) AS failed_attention,
        COUNT(*) FILTER (
            WHERE j.run_attempt_id IS NOT NULL
              AND j.state::TEXT IN (
                    'completed',
                    'skipped',
                    'failed',
                    'expired',
                    'cancelled'
                  )
              AND (
                    ja.terminal_accepted IS FALSE
                    OR (
                        ja.terminal_work_state IS NOT NULL
                        AND ja.terminal_work_state <> j.state::TEXT
                    )
                  )
        ) AS terminal_mismatch
    FROM {schema}.job AS j
    LEFT JOIN {schema}.job_attempt AS ja
      ON ja.run_attempt_id = j.run_attempt_id
    WHERE j.dag_id = p_dag_id
    GROUP BY j.dag_id
), recent_history AS (
    SELECT LOWER(COALESCE(state, 'created')) AS state, history_created_on
    FROM {schema}.dag_history
    WHERE id = p_dag_id
    ORDER BY history_created_on DESC
    LIMIT 32
), dag_lifecycle AS (
    SELECT
        JSONB_AGG(
            JSONB_BUILD_OBJECT(
                'state', state,
                'at', history_created_on
            )
            ORDER BY history_created_on
        ) AS events,
        (ARRAY_AGG(state ORDER BY history_created_on DESC))[1] AS latest_state
    FROM recent_history
)
SELECT
    d.id,
    d.name::TEXT,
    LOWER(COALESCE(d.state, 'created')),
    d.planner::TEXT,
    d.priority,
    d.task_count,
    d.created_on,
    d.started_on,
    d.completed_on,
    d.updated_on,
    EXTRACT(EPOCH FROM (NOW() - d.created_on))::DOUBLE PRECISION,
    EXTRACT(EPOCH FROM (
        NOW() - GREATEST(d.updated_on, COALESCE(js.last_updated_on, d.updated_on))
    ))::DOUBLE PRECISION,
    COALESCE(js.total, 0),
    COALESCE(js.created, 0),
    COALESCE(js.retry, 0),
    COALESCE(js.active, 0),
    COALESCE(js.completed, 0),
    COALESCE(js.skipped, 0),
    COALESCE(js.expired, 0),
    COALESCE(js.cancelled, 0),
    COALESCE(js.failed, 0),
    COALESCE(js.queues, ARRAY[]::TEXT[]),
    COALESCE(js.queued_too_long, 0),
    COALESCE(js.running_too_long, 0),
    COALESCE(js.stale_update, 0),
    COALESCE(js.retrying, 0),
    COALESCE(js.failed_attention, 0),
    COALESCE(js.terminal_mismatch, 0),
    CASE
        WHEN dl.events IS NULL THEN JSONB_BUILD_ARRAY(
            JSONB_BUILD_OBJECT(
                'state', LOWER(COALESCE(d.state, 'created')),
                'at', d.updated_on
            )
        )
        WHEN dl.latest_state IS DISTINCT FROM LOWER(COALESCE(d.state, 'created'))
            THEN dl.events || JSONB_BUILD_ARRAY(
                JSONB_BUILD_OBJECT(
                    'state', LOWER(COALESCE(d.state, 'created')),
                    'at', d.updated_on
                )
            )
        ELSE dl.events
    END
FROM {schema}.dag AS d
LEFT JOIN job_stats AS js ON js.dag_id = d.id
CROSS JOIN dag_lifecycle AS dl
WHERE d.id = p_dag_id;
$function$;

COMMENT ON FUNCTION {schema}.get_operational_dag(UUID, INTEGER, INTEGER, INTEGER)
IS 'Returns payload-free DAG state, job rollups, attention counts, and bounded lifecycle history.';
