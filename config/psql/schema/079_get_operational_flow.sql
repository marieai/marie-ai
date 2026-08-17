-- File: 079_get_operational_flow.sql
-- Description: Bounded scheduler flow-pressure snapshot
-- Dependencies: 005_job.sql, 006_job_history.sql, 065_job_attempt.sql

CREATE OR REPLACE FUNCTION {schema}.get_operational_flow(
    p_window_seconds INTEGER DEFAULT 900,
    p_queue TEXT DEFAULT NULL,
    p_queue_limit INTEGER DEFAULT 25
)
RETURNS TABLE (
    observed_at TIMESTAMPTZ,
    arrivals BIGINT,
    ready_transitions BIGINT,
    attempt_activations BIGINT,
    starts BIGINT,
    terminals BIGINT,
    failures BIGINT,
    ready_now BIGINT,
    active_now BIGINT,
    oldest_ready_seconds DOUBLE PRECISION,
    ready_to_running_p50_seconds DOUBLE PRECISION,
    ready_to_running_p95_seconds DOUBLE PRECISION,
    ready_to_running_max_seconds DOUBLE PRECISION,
    running_to_terminal_p50_seconds DOUBLE PRECISION,
    running_to_terminal_p95_seconds DOUBLE PRECISION,
    running_to_terminal_max_seconds DOUBLE PRECISION,
    queues JSONB
)
LANGUAGE SQL
STABLE
AS $function$
WITH arrivals AS (
    SELECT COUNT(*)::BIGINT AS count
    FROM {schema}.job AS j
    WHERE j.created_on >= NOW() - make_interval(secs => p_window_seconds)
      AND (p_queue IS NULL OR j.name = p_queue)
), ready_transitions AS (
    SELECT COUNT(*)::BIGINT AS count
    FROM {schema}.job_history AS jh
    WHERE jh.history_created_on >= NOW() - make_interval(secs => p_window_seconds)
      AND jh.state IN ('created', 'retry')
      AND (p_queue IS NULL OR jh.name = p_queue)
), activations AS (
    SELECT COUNT(*)::BIGINT AS count
    FROM {schema}.job_attempt AS ja
    WHERE ja.activated_at >= NOW() - make_interval(secs => p_window_seconds)
      AND (p_queue IS NULL OR ja.job_name = p_queue)
), starts AS (
    SELECT COUNT(*)::BIGINT AS count
    FROM {schema}.job AS j
    WHERE j.started_on >= NOW() - make_interval(secs => p_window_seconds)
      AND (p_queue IS NULL OR j.name = p_queue)
), terminals AS (
    SELECT
        COUNT(*)::BIGINT AS count,
        COUNT(*) FILTER (
            WHERE j.state IN ('failed', 'expired', 'cancelled')
        )::BIGINT AS failures
    FROM {schema}.job AS j
    WHERE j.completed_on >= NOW() - make_interval(secs => p_window_seconds)
      AND j.state IN ('completed', 'skipped', 'failed', 'expired', 'cancelled')
      AND (p_queue IS NULL OR j.name = p_queue)
), current_work AS (
    SELECT
        COUNT(*) FILTER (
            WHERE j.state IN ('created', 'retry') AND j.start_after <= NOW()
        )::BIGINT AS ready,
        COUNT(*) FILTER (WHERE j.state = 'active')::BIGINT AS active,
        MAX(EXTRACT(EPOCH FROM (NOW() - j.start_after))) FILTER (
            WHERE j.state IN ('created', 'retry') AND j.start_after <= NOW()
        )::DOUBLE PRECISION AS oldest_ready_seconds
    FROM {schema}.job AS j
    WHERE (p_queue IS NULL OR j.name = p_queue)
      AND j.state IN ('created', 'retry', 'active')
), stage_latency AS (
    SELECT
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (j.started_on - GREATEST(j.created_on, j.start_after)))
        ) FILTER (
            WHERE j.started_on >= NOW() - make_interval(secs => p_window_seconds)
        ) AS ready_p50,
        percentile_cont(0.95) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (j.started_on - GREATEST(j.created_on, j.start_after)))
        ) FILTER (
            WHERE j.started_on >= NOW() - make_interval(secs => p_window_seconds)
        ) AS ready_p95,
        MAX(EXTRACT(EPOCH FROM (j.started_on - GREATEST(j.created_on, j.start_after))))
            FILTER (
                WHERE j.started_on >= NOW() - make_interval(secs => p_window_seconds)
            ) AS ready_max,
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (j.completed_on - j.started_on))
        ) FILTER (
            WHERE j.completed_on >= NOW() - make_interval(secs => p_window_seconds)
              AND j.started_on IS NOT NULL
        ) AS run_p50,
        percentile_cont(0.95) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (j.completed_on - j.started_on))
        ) FILTER (
            WHERE j.completed_on >= NOW() - make_interval(secs => p_window_seconds)
              AND j.started_on IS NOT NULL
        ) AS run_p95,
        MAX(EXTRACT(EPOCH FROM (j.completed_on - j.started_on)))
            FILTER (
                WHERE j.completed_on >= NOW() - make_interval(secs => p_window_seconds)
                  AND j.started_on IS NOT NULL
            ) AS run_max
    FROM {schema}.job AS j
    WHERE (p_queue IS NULL OR j.name = p_queue)
      AND (
          j.started_on >= NOW() - make_interval(secs => p_window_seconds)
          OR j.completed_on >= NOW() - make_interval(secs => p_window_seconds)
      )
), queue_counts AS (
    SELECT
        j.name,
        COUNT(*) FILTER (
            WHERE j.created_on >= NOW() - make_interval(secs => p_window_seconds)
        )::BIGINT AS arrivals,
        COUNT(*) FILTER (
            WHERE j.completed_on >= NOW() - make_interval(secs => p_window_seconds)
              AND j.state IN ('completed', 'skipped', 'failed', 'expired', 'cancelled')
        )::BIGINT AS terminals,
        COUNT(*) FILTER (
            WHERE j.completed_on >= NOW() - make_interval(secs => p_window_seconds)
              AND j.state IN ('failed', 'expired', 'cancelled')
        )::BIGINT AS failures,
        COUNT(*) FILTER (
            WHERE j.state IN ('created', 'retry') AND j.start_after <= NOW()
        )::BIGINT AS ready,
        COUNT(*) FILTER (WHERE j.state = 'active')::BIGINT AS active,
        MAX(EXTRACT(EPOCH FROM (NOW() - j.start_after))) FILTER (
            WHERE j.state IN ('created', 'retry') AND j.start_after <= NOW()
        )::DOUBLE PRECISION AS oldest_ready_seconds
    FROM {schema}.job AS j
    WHERE (p_queue IS NULL OR j.name = p_queue)
      AND (
          j.created_on >= NOW() - make_interval(secs => p_window_seconds)
          OR j.completed_on >= NOW() - make_interval(secs => p_window_seconds)
          OR j.state IN ('created', 'retry', 'active')
      )
    GROUP BY j.name
), queue_page AS (
    SELECT *
    FROM queue_counts
    ORDER BY
        ABS(arrivals - terminals) DESC,
        ready DESC,
        name
    LIMIT LEAST(GREATEST(p_queue_limit, 1), 100)
), queue_json AS (
    SELECT COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'name', name,
                'arrivals', arrivals,
                'terminals', terminals,
                'failures', failures,
                'ready', ready,
                'active', active,
                'oldest_ready_seconds', oldest_ready_seconds
            )
            ORDER BY ABS(arrivals - terminals) DESC, ready DESC, name
        ),
        '[]'::JSONB
    ) AS queues
    FROM queue_page
)
SELECT
    NOW() AS observed_at,
    arrivals.count,
    ready_transitions.count,
    activations.count,
    starts.count,
    terminals.count,
    terminals.failures,
    current_work.ready,
    current_work.active,
    current_work.oldest_ready_seconds,
    stage_latency.ready_p50::DOUBLE PRECISION,
    stage_latency.ready_p95::DOUBLE PRECISION,
    stage_latency.ready_max::DOUBLE PRECISION,
    stage_latency.run_p50::DOUBLE PRECISION,
    stage_latency.run_p95::DOUBLE PRECISION,
    stage_latency.run_max::DOUBLE PRECISION,
    queue_json.queues
FROM arrivals
CROSS JOIN ready_transitions
CROSS JOIN activations
CROSS JOIN starts
CROSS JOIN terminals
CROSS JOIN current_work
CROSS JOIN stage_latency
CROSS JOIN queue_json;
$function$;

COMMENT ON FUNCTION {schema}.get_operational_flow(INTEGER, TEXT, INTEGER)
IS 'Returns bounded scheduler flow rates, backlog pressure, stage latency, and queue breakdowns.';
