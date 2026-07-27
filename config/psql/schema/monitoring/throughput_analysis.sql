-- Install parameterized scheduler-throughput functions.
--
-- Units:
--   plan throughput          completed DAGs (one DAG is one query-plan run)
--   task throughput          completed jobs (one job is one query-plan task)
--   executor task throughput completed jobs excluding scheduler-local control nodes
--
-- Install this file once, then call the functions with ordinary SQL:
--   SELECT * FROM marie_scheduler.monitor_system_throughput(24, NULL);
--   SELECT * FROM marie_scheduler.monitor_planner_throughput(24, NULL);
--   SELECT * FROM marie_scheduler.monitor_task_throughput(24, NULL);

-- Overall and hourly system throughput. The window_total row contains average
-- hourly rates. Hour rows contain actual counts; the current hour is marked as
-- partial and should not be compared directly with a completed clock hour.
CREATE OR REPLACE FUNCTION marie_scheduler.monitor_system_throughput(
    p_lookback_hours integer DEFAULT 24,
    p_planner_name text DEFAULT NULL
)
RETURNS TABLE (
    period text,
    period_start_utc timestamptz,
    period_end_utc timestamptz,
    partial boolean,
    plans_submitted bigint,
    plans_completed bigint,
    plans_failed bigint,
    plans_expired bigint,
    plans_cancelled bigint,
    plan_success_rate_pct numeric,
    tasks_completed bigint,
    executor_tasks_completed bigint,
    tasks_failed bigint,
    tasks_expired bigint,
    tasks_cancelled bigint,
    tasks_skipped bigint,
    task_success_rate_pct numeric,
    avg_completed_plans_per_hour numeric,
    avg_completed_executor_tasks_per_hour numeric
)
LANGUAGE sql
STABLE
AS $function$
WITH params AS (
    SELECT
        LEAST(
            GREATEST(COALESCE(p_lookback_hours, 24), 1),
            720
        ) AS lookback_hours,
        NULLIF(BTRIM(p_planner_name), '') AS planner_name
), bounds AS (
    SELECT
        date_trunc('hour', NOW(), 'UTC')
            - make_interval(hours => p.lookback_hours - 1) AS window_start,
        NOW() AS observed_at,
        date_trunc('hour', NOW(), 'UTC') + INTERVAL '1 hour' AS bucket_end,
        p.planner_name
    FROM params p
), hours AS (
    SELECT generate_series(
        b.window_start,
        b.bucket_end - INTERVAL '1 hour',
        INTERVAL '1 hour'
    ) AS bucket_start
    FROM bounds b
), events AS (
    SELECT
        date_trunc('hour', d.created_on, 'UTC') AS bucket_start,
        1::bigint AS plans_submitted,
        0::bigint AS plans_completed,
        0::bigint AS plans_failed,
        0::bigint AS plans_expired,
        0::bigint AS plans_cancelled,
        0::bigint AS tasks_completed,
        0::bigint AS executor_tasks_completed,
        0::bigint AS tasks_failed,
        0::bigint AS tasks_expired,
        0::bigint AS tasks_cancelled,
        0::bigint AS tasks_skipped
    FROM marie_scheduler.dag d
    CROSS JOIN bounds b
    WHERE d.created_on >= b.window_start
      AND d.created_on < b.observed_at
      AND (b.planner_name IS NULL OR d.planner = b.planner_name)

    UNION ALL

    SELECT
        date_trunc('hour', d.completed_on, 'UTC'),
        0,
        CASE WHEN d.state = 'completed' THEN 1 ELSE 0 END,
        CASE WHEN d.state = 'failed' THEN 1 ELSE 0 END,
        CASE WHEN d.state = 'expired' THEN 1 ELSE 0 END,
        CASE WHEN d.state = 'cancelled' THEN 1 ELSE 0 END,
        0,
        0,
        0,
        0,
        0,
        0
    FROM marie_scheduler.dag d
    CROSS JOIN bounds b
    WHERE d.completed_on >= b.window_start
      AND d.completed_on < b.observed_at
      AND d.state IN ('completed', 'failed', 'cancelled', 'expired')
      AND (b.planner_name IS NULL OR d.planner = b.planner_name)

    UNION ALL

    SELECT
        date_trunc('hour', j.completed_on, 'UTC'),
        0,
        0,
        0,
        0,
        0,
        CASE WHEN j.state::text = 'completed' THEN 1 ELSE 0 END,
        CASE
            WHEN j.state::text = 'completed'
             AND COALESCE(j.data #>> '{metadata,on}', '') NOT IN (
                 '',
                 'noop://noop',
                 'branch://control',
                 'switch://control',
                 'merger://control'
             )
            THEN 1
            ELSE 0
        END,
        CASE WHEN j.state::text = 'failed' THEN 1 ELSE 0 END,
        CASE WHEN j.state::text = 'expired' THEN 1 ELSE 0 END,
        CASE WHEN j.state::text = 'cancelled' THEN 1 ELSE 0 END,
        CASE WHEN j.state::text = 'skipped' THEN 1 ELSE 0 END
    FROM marie_scheduler.job j
    JOIN marie_scheduler.dag d ON d.id = j.dag_id
    CROSS JOIN bounds b
    WHERE j.completed_on >= b.window_start
      AND j.completed_on < b.observed_at
      AND j.state::text IN (
          'completed',
          'failed',
          'expired',
          'cancelled',
          'skipped'
      )
      AND (b.planner_name IS NULL OR d.planner = b.planner_name)
), hourly AS (
    SELECT
        h.bucket_start,
        COALESCE(SUM(e.plans_submitted), 0)::bigint AS plans_submitted,
        COALESCE(SUM(e.plans_completed), 0)::bigint AS plans_completed,
        COALESCE(SUM(e.plans_failed), 0)::bigint AS plans_failed,
        COALESCE(SUM(e.plans_expired), 0)::bigint AS plans_expired,
        COALESCE(SUM(e.plans_cancelled), 0)::bigint AS plans_cancelled,
        COALESCE(SUM(e.tasks_completed), 0)::bigint AS tasks_completed,
        COALESCE(SUM(e.executor_tasks_completed), 0)::bigint
            AS executor_tasks_completed,
        COALESCE(SUM(e.tasks_failed), 0)::bigint AS tasks_failed,
        COALESCE(SUM(e.tasks_expired), 0)::bigint AS tasks_expired,
        COALESCE(SUM(e.tasks_cancelled), 0)::bigint AS tasks_cancelled,
        COALESCE(SUM(e.tasks_skipped), 0)::bigint AS tasks_skipped
    FROM hours h
    LEFT JOIN events e ON e.bucket_start = h.bucket_start
    GROUP BY h.bucket_start
), output AS (
    SELECT
        0 AS sort_order,
        'window_total'::text AS period,
        b.window_start AS period_start_utc,
        b.observed_at AS period_end_utc,
        FALSE AS partial,
        SUM(h.plans_submitted)::bigint AS plans_submitted,
        SUM(h.plans_completed)::bigint AS plans_completed,
        SUM(h.plans_failed)::bigint AS plans_failed,
        SUM(h.plans_expired)::bigint AS plans_expired,
        SUM(h.plans_cancelled)::bigint AS plans_cancelled,
        ROUND(
            100.0 * SUM(h.plans_completed)
            / NULLIF(SUM(
                h.plans_completed
                + h.plans_failed
                + h.plans_expired
                + h.plans_cancelled
            ), 0),
            2
        ) AS plan_success_rate_pct,
        SUM(h.tasks_completed)::bigint AS tasks_completed,
        SUM(h.executor_tasks_completed)::bigint AS executor_tasks_completed,
        SUM(h.tasks_failed)::bigint AS tasks_failed,
        SUM(h.tasks_expired)::bigint AS tasks_expired,
        SUM(h.tasks_cancelled)::bigint AS tasks_cancelled,
        SUM(h.tasks_skipped)::bigint AS tasks_skipped,
        ROUND(
            100.0 * SUM(h.tasks_completed)
            / NULLIF(SUM(
                h.tasks_completed
                + h.tasks_failed
                + h.tasks_expired
                + h.tasks_cancelled
            ), 0),
            2
        ) AS task_success_rate_pct,
        ROUND(
            SUM(h.plans_completed)
            / NULLIF(
                EXTRACT(EPOCH FROM (b.observed_at - b.window_start))
                    / 3600.0,
                0
            ),
            2
        ) AS avg_completed_plans_per_hour,
        ROUND(
            SUM(h.executor_tasks_completed)
            / NULLIF(
                EXTRACT(EPOCH FROM (b.observed_at - b.window_start))
                    / 3600.0,
                0
            ),
            2
        ) AS avg_completed_executor_tasks_per_hour
    FROM hourly h
    CROSS JOIN bounds b
    GROUP BY b.window_start, b.observed_at

    UNION ALL

    SELECT
        1,
        'hour',
        h.bucket_start,
        LEAST(h.bucket_start + INTERVAL '1 hour', b.observed_at),
        h.bucket_start = date_trunc('hour', b.observed_at, 'UTC'),
        h.plans_submitted,
        h.plans_completed,
        h.plans_failed,
        h.plans_expired,
        h.plans_cancelled,
        ROUND(
            100.0 * h.plans_completed
            / NULLIF(
                h.plans_completed
                + h.plans_failed
                + h.plans_expired
                + h.plans_cancelled,
                0
            ),
            2
        ),
        h.tasks_completed,
        h.executor_tasks_completed,
        h.tasks_failed,
        h.tasks_expired,
        h.tasks_cancelled,
        h.tasks_skipped,
        ROUND(
            100.0 * h.tasks_completed
            / NULLIF(
                h.tasks_completed
                + h.tasks_failed
                + h.tasks_expired
                + h.tasks_cancelled,
                0
            ),
            2
        ),
        NULL::numeric,
        NULL::numeric
    FROM hourly h
    CROSS JOIN bounds b
)
SELECT
    o.period,
    o.period_start_utc,
    o.period_end_utc,
    o.partial,
    o.plans_submitted,
    o.plans_completed,
    o.plans_failed,
    o.plans_expired,
    o.plans_cancelled,
    o.plan_success_rate_pct,
    o.tasks_completed,
    o.executor_tasks_completed,
    o.tasks_failed,
    o.tasks_expired,
    o.tasks_cancelled,
    o.tasks_skipped,
    o.task_success_rate_pct,
    o.avg_completed_plans_per_hour,
    o.avg_completed_executor_tasks_per_hour
FROM output o
ORDER BY o.sort_order, o.period_start_utc;
$function$;

-- Window-total and hourly throughput by query planner. Missing hours are
-- omitted for planners with no activity in that hour.
CREATE OR REPLACE FUNCTION marie_scheduler.monitor_planner_throughput(
    p_lookback_hours integer DEFAULT 24,
    p_planner_name text DEFAULT NULL
)
RETURNS TABLE (
    period text,
    bucket_start_utc timestamptz,
    planner text,
    plans_submitted bigint,
    plans_completed bigint,
    plans_failed bigint,
    plans_expired bigint,
    plans_cancelled bigint,
    executor_tasks_completed bigint,
    tasks_failed bigint,
    tasks_expired bigint,
    tasks_cancelled bigint
)
LANGUAGE sql
STABLE
AS $function$
WITH params AS (
    SELECT
        LEAST(
            GREATEST(COALESCE(p_lookback_hours, 24), 1),
            720
        ) AS lookback_hours,
        NULLIF(BTRIM(p_planner_name), '') AS planner_name
), bounds AS (
    SELECT
        date_trunc('hour', NOW(), 'UTC')
            - make_interval(hours => p.lookback_hours - 1) AS window_start,
        NOW() AS observed_at,
        p.planner_name
    FROM params p
), events AS (
    SELECT
        date_trunc('hour', d.created_on, 'UTC') AS bucket_start_utc,
        COALESCE(d.planner::text, '[unknown]') AS planner,
        1::bigint AS plans_submitted,
        0::bigint AS plans_completed,
        0::bigint AS plans_failed,
        0::bigint AS plans_expired,
        0::bigint AS plans_cancelled,
        0::bigint AS executor_tasks_completed,
        0::bigint AS tasks_failed,
        0::bigint AS tasks_expired,
        0::bigint AS tasks_cancelled
    FROM marie_scheduler.dag d
    CROSS JOIN bounds b
    WHERE d.created_on >= b.window_start
      AND d.created_on < b.observed_at
      AND (b.planner_name IS NULL OR d.planner = b.planner_name)

    UNION ALL

    SELECT
        date_trunc('hour', d.completed_on, 'UTC'),
        COALESCE(d.planner::text, '[unknown]'),
        0,
        CASE WHEN d.state = 'completed' THEN 1 ELSE 0 END,
        CASE WHEN d.state = 'failed' THEN 1 ELSE 0 END,
        CASE WHEN d.state = 'expired' THEN 1 ELSE 0 END,
        CASE WHEN d.state = 'cancelled' THEN 1 ELSE 0 END,
        0,
        0,
        0,
        0
    FROM marie_scheduler.dag d
    CROSS JOIN bounds b
    WHERE d.completed_on >= b.window_start
      AND d.completed_on < b.observed_at
      AND d.state IN ('completed', 'failed', 'cancelled', 'expired')
      AND (b.planner_name IS NULL OR d.planner = b.planner_name)

    UNION ALL

    SELECT
        date_trunc('hour', j.completed_on, 'UTC'),
        COALESCE(d.planner::text, '[unknown]'),
        0,
        0,
        0,
        0,
        0,
        CASE
            WHEN j.state::text = 'completed'
             AND COALESCE(j.data #>> '{metadata,on}', '') NOT IN (
                 '',
                 'noop://noop',
                 'branch://control',
                 'switch://control',
                 'merger://control'
             )
            THEN 1
            ELSE 0
        END,
        CASE WHEN j.state::text = 'failed' THEN 1 ELSE 0 END,
        CASE WHEN j.state::text = 'expired' THEN 1 ELSE 0 END,
        CASE WHEN j.state::text = 'cancelled' THEN 1 ELSE 0 END
    FROM marie_scheduler.job j
    JOIN marie_scheduler.dag d ON d.id = j.dag_id
    CROSS JOIN bounds b
    WHERE j.completed_on >= b.window_start
      AND j.completed_on < b.observed_at
      AND j.state::text IN ('completed', 'failed', 'expired', 'cancelled')
      AND (b.planner_name IS NULL OR d.planner = b.planner_name)
), grouped AS (
SELECT
    CASE
        WHEN GROUPING(e.bucket_start_utc) = 1 THEN 'window_total'
        ELSE 'hour'
    END AS period,
    e.bucket_start_utc,
    e.planner,
    SUM(e.plans_submitted)::bigint AS plans_submitted,
    SUM(e.plans_completed)::bigint AS plans_completed,
    SUM(e.plans_failed)::bigint AS plans_failed,
    SUM(e.plans_expired)::bigint AS plans_expired,
    SUM(e.plans_cancelled)::bigint AS plans_cancelled,
    SUM(e.executor_tasks_completed)::bigint AS executor_tasks_completed,
    SUM(e.tasks_failed)::bigint AS tasks_failed,
    SUM(e.tasks_expired)::bigint AS tasks_expired,
    SUM(e.tasks_cancelled)::bigint AS tasks_cancelled,
    GROUPING(e.bucket_start_utc) AS bucket_group
FROM events e
GROUP BY GROUPING SETS (
    (e.planner),
    (e.bucket_start_utc, e.planner)
)
)
SELECT
    g.period,
    g.bucket_start_utc,
    g.planner,
    g.plans_submitted,
    g.plans_completed,
    g.plans_failed,
    g.plans_expired,
    g.plans_cancelled,
    g.executor_tasks_completed,
    g.tasks_failed,
    g.tasks_expired,
    g.tasks_cancelled
FROM grouped g
ORDER BY g.bucket_group DESC, g.planner, g.bucket_start_utc;
$function$;

-- Window-total and hourly query-plan task throughput. Task name comes from
-- metadata.name and endpoint comes from metadata.on; queue name is included
-- to expose routing.
CREATE OR REPLACE FUNCTION marie_scheduler.monitor_task_throughput(
    p_lookback_hours integer DEFAULT 24,
    p_planner_name text DEFAULT NULL
)
RETURNS TABLE (
    period text,
    bucket_start_utc timestamptz,
    planner text,
    queue_name text,
    task_name text,
    endpoint text,
    executor_backed boolean,
    tasks_completed bigint,
    tasks_failed bigint,
    tasks_expired bigint,
    tasks_cancelled bigint,
    tasks_skipped bigint,
    avg_execution_seconds numeric,
    p95_execution_seconds numeric
)
LANGUAGE sql
STABLE
AS $function$
WITH params AS (
    SELECT
        LEAST(
            GREATEST(COALESCE(p_lookback_hours, 24), 1),
            720
        ) AS lookback_hours,
        NULLIF(BTRIM(p_planner_name), '') AS planner_name
), bounds AS (
    SELECT
        date_trunc('hour', NOW(), 'UTC')
            - make_interval(hours => p.lookback_hours - 1) AS window_start,
        NOW() AS observed_at,
        p.planner_name
    FROM params p
), task_events AS (
    SELECT
        date_trunc('hour', j.completed_on, 'UTC') AS bucket_start_utc,
        COALESCE(d.planner::text, '[unknown]') AS planner,
        j.name::text AS queue_name,
        COALESCE(
            NULLIF(j.data #>> '{metadata,name}', ''),
            '[unnamed task]'
        ) AS task_name,
        COALESCE(
            NULLIF(j.data #>> '{metadata,on}', ''),
            '[no endpoint]'
        ) AS endpoint,
        COALESCE(j.data #>> '{metadata,on}', '') NOT IN (
            '',
            'noop://noop',
            'branch://control',
            'switch://control',
            'merger://control'
        ) AS executor_backed,
        j.state::text AS terminal_state,
        EXTRACT(EPOCH FROM (j.completed_on - j.started_on))
            AS execution_seconds
    FROM marie_scheduler.job j
    JOIN marie_scheduler.dag d ON d.id = j.dag_id
    CROSS JOIN bounds b
    WHERE j.completed_on >= b.window_start
      AND j.completed_on < b.observed_at
      AND j.state::text IN (
          'completed',
          'failed',
          'expired',
          'cancelled',
          'skipped'
      )
      AND (b.planner_name IS NULL OR d.planner = b.planner_name)
), grouped AS (
SELECT
    CASE
        WHEN GROUPING(t.bucket_start_utc) = 1 THEN 'window_total'
        ELSE 'hour'
    END AS period,
    t.bucket_start_utc,
    t.planner,
    t.queue_name,
    t.task_name,
    t.endpoint,
    t.executor_backed,
    COUNT(*) FILTER (
        WHERE t.terminal_state = 'completed'
    ) AS tasks_completed,
    COUNT(*) FILTER (
        WHERE t.terminal_state = 'failed'
    ) AS tasks_failed,
    COUNT(*) FILTER (
        WHERE t.terminal_state = 'expired'
    ) AS tasks_expired,
    COUNT(*) FILTER (
        WHERE t.terminal_state = 'cancelled'
    ) AS tasks_cancelled,
    COUNT(*) FILTER (
        WHERE t.terminal_state = 'skipped'
    ) AS tasks_skipped,
    ROUND(AVG(t.execution_seconds) FILTER (
        WHERE t.terminal_state = 'completed'
    )::numeric, 3) AS avg_execution_seconds,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (
        ORDER BY t.execution_seconds
    ) FILTER (
        WHERE t.terminal_state = 'completed'
          AND t.execution_seconds IS NOT NULL
    )::numeric, 3) AS p95_execution_seconds,
    GROUPING(t.bucket_start_utc) AS bucket_group
FROM task_events t
GROUP BY GROUPING SETS (
    (
        t.planner,
        t.queue_name,
        t.task_name,
        t.endpoint,
        t.executor_backed
    ),
    (
        t.bucket_start_utc,
        t.planner,
        t.queue_name,
        t.task_name,
        t.endpoint,
        t.executor_backed
    )
)
)
SELECT
    g.period,
    g.bucket_start_utc,
    g.planner,
    g.queue_name,
    g.task_name,
    g.endpoint,
    g.executor_backed,
    g.tasks_completed,
    g.tasks_failed,
    g.tasks_expired,
    g.tasks_cancelled,
    g.tasks_skipped,
    g.avg_execution_seconds,
    g.p95_execution_seconds
FROM grouped g
ORDER BY
    g.bucket_group DESC,
    g.planner,
    g.queue_name,
    g.task_name,
    g.endpoint,
    g.bucket_start_utc;
$function$;
