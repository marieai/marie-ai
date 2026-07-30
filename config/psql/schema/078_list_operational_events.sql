-- File: 078_list_operational_events.sql
-- Description: Cursor-paged operational lifecycle events
-- Dependencies: 006_job_history.sql, 008_dag_history.sql, 065_job_attempt.sql

CREATE OR REPLACE FUNCTION {schema}.list_operational_events(
    p_limit INTEGER DEFAULT 25,
    p_before_at TIMESTAMPTZ DEFAULT NULL,
    p_before_id TEXT DEFAULT NULL,
    p_window_seconds INTEGER DEFAULT 900,
    p_severity TEXT DEFAULT NULL,
    p_component TEXT DEFAULT NULL,
    p_search TEXT DEFAULT NULL
)
RETURNS TABLE (
    event_id TEXT,
    occurred_at TIMESTAMPTZ,
    severity TEXT,
    component TEXT,
    event_code TEXT,
    affected_type TEXT,
    affected_id TEXT,
    job_id UUID,
    dag_id UUID,
    run_attempt_id UUID,
    executor TEXT,
    gateway_instance_id TEXT,
    summary TEXT
)
LANGUAGE SQL
STABLE
AS $function$
WITH events AS NOT MATERIALIZED (
    SELECT
        'job:' || jh.history_id::TEXT AS event_id,
        jh.history_created_on AS occurred_at,
        CASE
            WHEN jh.state::TEXT IN ('failed', 'expired', 'cancelled') THEN 'bad'
            WHEN jh.state::TEXT = 'retry' THEN 'warning'
            ELSE 'info'
        END AS severity,
        'scheduler.job'::TEXT AS component,
        'JOB_' || UPPER(jh.state::TEXT) AS event_code,
        'job'::TEXT AS affected_type,
        jh.id::TEXT AS affected_id,
        jh.id AS job_id,
        jh.dag_id,
        jh.run_attempt_id,
        NULL::TEXT AS executor,
        NULL::TEXT AS gateway_instance_id,
        'Job state changed to ' || LOWER(jh.state::TEXT) AS summary
    FROM {schema}.job_history AS jh
    WHERE jh.history_created_on >= NOW() - make_interval(secs => p_window_seconds)

    UNION ALL

    SELECT
        'dag:' || dh.history_id::TEXT,
        dh.history_created_on,
        CASE
            WHEN LOWER(COALESCE(dh.state, 'created')) IN ('failed', 'expired', 'cancelled') THEN 'bad'
            ELSE 'info'
        END,
        'scheduler.dag',
        'DAG_' || UPPER(COALESCE(dh.state, 'created')),
        'dag',
        dh.id::TEXT,
        NULL::UUID,
        dh.id,
        NULL::UUID,
        NULL::TEXT,
        NULL::TEXT,
        'DAG state changed to ' || LOWER(COALESCE(dh.state, 'created'))
    FROM {schema}.dag_history AS dh
    WHERE dh.history_created_on >= NOW() - make_interval(secs => p_window_seconds)

    UNION ALL

    SELECT
        'attempt:' || ja.run_attempt_id::TEXT || ':activated',
        ja.activated_at,
        'info',
        'scheduler.attempt',
        'ATTEMPT_ACTIVATED',
        'attempt',
        ja.run_attempt_id::TEXT,
        ja.job_id,
        ja.dag_id,
        ja.run_attempt_id,
        ja.executor,
        ja.gateway_instance_id,
        'Execution attempt activated'
    FROM {schema}.job_attempt AS ja
    WHERE ja.activated_at >= NOW() - make_interval(secs => p_window_seconds)

    UNION ALL

    SELECT
        'attempt:' || ja.run_attempt_id::TEXT || ':terminal',
        ja.terminal_at,
        CASE
            WHEN ja.terminal_accepted IS FALSE
              OR LOWER(COALESCE(ja.terminal_work_state, ja.terminal_status, ''))
                    IN ('failed', 'expired', 'cancelled')
            THEN 'bad'
            ELSE 'info'
        END,
        'scheduler.attempt',
        CASE
            WHEN ja.terminal_accepted IS FALSE THEN 'ATTEMPT_TERMINAL_REJECTED'
            ELSE 'ATTEMPT_' || UPPER(COALESCE(ja.terminal_work_state, ja.terminal_status, 'TERMINAL'))
        END,
        'attempt',
        ja.run_attempt_id::TEXT,
        ja.job_id,
        ja.dag_id,
        ja.run_attempt_id,
        ja.executor,
        COALESCE(ja.terminal_gateway_instance_id, ja.gateway_instance_id),
        CASE
            WHEN ja.terminal_accepted IS FALSE THEN 'Execution terminal update was rejected'
            ELSE 'Execution attempt reached a terminal state'
        END
    FROM {schema}.job_attempt AS ja
    WHERE ja.terminal_at IS NOT NULL
      AND ja.terminal_at >= NOW() - make_interval(secs => p_window_seconds)

    UNION ALL

    SELECT
        'attempt:' || ja.run_attempt_id::TEXT || ':recovery',
        ja.recovery_at,
        'warning',
        'scheduler.attempt',
        'ATTEMPT_RECOVERED',
        'attempt',
        ja.run_attempt_id::TEXT,
        ja.job_id,
        ja.dag_id,
        ja.run_attempt_id,
        ja.executor,
        ja.gateway_instance_id,
        'Execution attempt entered recovery'
    FROM {schema}.job_attempt AS ja
    WHERE ja.recovery_at IS NOT NULL
      AND ja.recovery_at >= NOW() - make_interval(secs => p_window_seconds)
), filtered AS NOT MATERIALIZED (
    SELECT *
    FROM events
    WHERE (p_severity IS NULL OR severity = p_severity)
      AND (p_component IS NULL OR component = p_component)
      AND (
          p_search IS NULL
          OR event_id ILIKE '%' || p_search || '%'
          OR event_code ILIKE '%' || p_search || '%'
          OR affected_id ILIKE '%' || p_search || '%'
          OR COALESCE(job_id::TEXT, '') ILIKE '%' || p_search || '%'
          OR COALESCE(dag_id::TEXT, '') ILIKE '%' || p_search || '%'
          OR COALESCE(run_attempt_id::TEXT, '') ILIKE '%' || p_search || '%'
          OR COALESCE(executor, '') ILIKE '%' || p_search || '%'
          OR COALESCE(gateway_instance_id, '') ILIKE '%' || p_search || '%'
      )
      AND (
          p_before_at IS NULL
          OR (occurred_at, event_id) < (
              p_before_at,
              COALESCE(p_before_id, repeat('~', 128))
          )
      )
)
SELECT *
FROM filtered
ORDER BY occurred_at DESC, event_id DESC
LIMIT LEAST(GREATEST(p_limit, 1), 100) + 1;
$function$;

COMMENT ON FUNCTION {schema}.list_operational_events(
    INTEGER, TIMESTAMPTZ, TEXT, INTEGER, TEXT, TEXT, TEXT
)
IS 'Returns cursor-paged, payload-free scheduler lifecycle events.';
