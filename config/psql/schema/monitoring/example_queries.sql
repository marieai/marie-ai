-- =========================================================
-- Marie Scheduler - Monitoring & Analytics Queries
-- =========================================================
-- Select and run one bounded query at a time. Do not execute this complete
-- catalog as a recurring production workload.
--
-- These queries use the generated slot columns to analyze
-- job patterns, capacity needs, SLA compliance, and load distribution.
--
-- Prerequisites:
--   - config/psql/schema/040_slots_columns.sql has been applied
--   - Job table has the generated slot/day columns
--
-- These are creation-time cohort and SLA analytics. They do not measure
-- completion throughput or historical concurrency. Use throughput_analysis.sql
-- for hourly query-plan and task throughput.
--
-- IMPORTANT: All data is stored in UTC timezone.
-- All slot calculations (slot_idx15_*, day_local_*) use UTC.
-- Despite the "local" name, these are UTC values - the name indicates
-- the conceptual purpose (local day/time for a timestamp).
-- For display in a specific timezone (e.g., America/Chicago),
-- use AT TIME ZONE in your queries. See examples below.
-- =========================================================


-- =========================================================
-- 1. HEATMAPS: Job Creation Patterns
-- =========================================================

-- 1.1 Daily Heatmap: Jobs Created by 15-min Time Slot
-- Purpose: Visualize when jobs are most commonly created throughout the day
-- Output: 96 rows (one per 15-min slot), ready for heatmap visualization
-- =========================================================
SELECT
    slot_idx15_created,
    LPAD((slot_idx15_created / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15_created % 4) * 15)::text, 2, '0') AS time_label,
    COUNT(*) AS total_jobs,
    COUNT(*) FILTER (WHERE state = 'completed') AS completed_jobs,
    COUNT(*) FILTER (WHERE state = 'failed') AS failed_jobs,
    COUNT(*) FILTER (
        WHERE state IN ('created', 'retry', 'active')
    ) AS outstanding_jobs,
    ROUND(100.0 * COUNT(*) FILTER (WHERE state = 'completed') / NULLIF(COUNT(*), 0), 1) AS completion_rate_pct
FROM marie_scheduler.job
WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 7
  AND slot_idx15_created IS NOT NULL
GROUP BY slot_idx15_created
ORDER BY slot_idx15_created;


-- 1.2 Weekly Heatmap: Day of Week × Time of Day
-- Purpose: Show patterns across different days of the week
-- Output: Matrix suitable for 2D heatmap (day × time slot)
-- =========================================================
SELECT
    EXTRACT(DOW FROM day_local_created) AS day_of_week,
    CASE EXTRACT(DOW FROM day_local_created)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS day_name,
    slot_idx15_created,
    LPAD((slot_idx15_created / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15_created % 4) * 15)::text, 2, '0') AS time_label,
    COUNT(*) AS job_count
FROM marie_scheduler.job
WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 30
  AND slot_idx15_created IS NOT NULL
GROUP BY day_of_week, day_name, slot_idx15_created
ORDER BY day_of_week, slot_idx15_created;


-- 1.3 Creation-Hour Cohort: Simplified View
-- Purpose: Group retained jobs by UTC hour of creation, not completion time
-- Output: 24 rows (one per hour)
-- =========================================================
SELECT
    (slot_idx15_created / 4) AS hour,
    LPAD((slot_idx15_created / 4)::text, 2, '0') || ':00' AS hour_label,
    COUNT(*) AS total_jobs,
    ROUND(AVG(EXTRACT(EPOCH FROM duration)), 2) AS avg_duration_seconds,
    COUNT(*) FILTER (WHERE state = 'completed') AS completed_count,
    COUNT(*) FILTER (WHERE state = 'failed') AS failed_count
FROM marie_scheduler.job
WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 7
  AND slot_idx15_created IS NOT NULL
GROUP BY hour
ORDER BY hour;


-- 1.4 SLA Deadline Heatmap: When are SLAs Due?
-- Purpose: Visualize when soft/hard SLA deadlines cluster
-- Output: Shows pressure points in the day
-- =========================================================
SELECT
    slot_idx15_soft,
    LPAD((slot_idx15_soft / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15_soft % 4) * 15)::text, 2, '0') AS soft_sla_time,
    COUNT(*) AS jobs_with_soft_sla,
    COUNT(*) FILTER (
        WHERE completed_on IS NOT NULL AND completed_on > soft_sla
    ) AS soft_sla_missed,
    COUNT(*) FILTER (
        WHERE state IN ('created', 'retry', 'active')
          AND now() > soft_sla
    ) AS soft_overdue_pending
FROM marie_scheduler.job
WHERE day_local_soft >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 7
  AND soft_sla IS NOT NULL
  AND slot_idx15_soft IS NOT NULL
GROUP BY slot_idx15_soft
ORDER BY slot_idx15_soft;


-- =========================================================
-- 2. CAPACITY PLANNING: Resource Allocation by Time Slot
-- =========================================================

-- 2.1 Current Workload by Effective Time Slot
-- Purpose: Group current job states by their effective SLA/creation slot
-- Output: Current-state cohorts, not historical concurrency
-- =========================================================
WITH workload_by_slot AS (
    SELECT
        slot_idx15_effective,
        LPAD((slot_idx15_effective / 4)::text, 2, '0') || ':' ||
        LPAD(((slot_idx15_effective % 4) * 15)::text, 2, '0') AS time_slot,
        COUNT(*) FILTER (WHERE state = 'active') AS active_count,
        COUNT(*) FILTER (WHERE state IN ('created', 'retry')) AS pending_count,
        COUNT(*) AS total_workload,
        ROUND(AVG(priority), 2) AS avg_priority
    FROM marie_scheduler.job
    WHERE day_local_effective >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 7
      AND slot_idx15_effective IS NOT NULL
      AND state IN ('created', 'retry', 'active')
    GROUP BY slot_idx15_effective
)
SELECT
    slot_idx15_effective,
    time_slot,
    active_count,
    pending_count,
    total_workload,
    avg_priority,
    -- Rank by total workload to identify peak periods
    RANK() OVER (ORDER BY total_workload DESC) AS load_rank
FROM workload_by_slot
ORDER BY total_workload DESC
LIMIT 20;


-- 2.2 Capacity Requirements: Recommended Resources per Time Slot
-- Purpose: Calculate how many workers/resources needed per time slot
-- Output: Resource allocation recommendations
-- =========================================================
WITH slot_metrics AS (
    SELECT
        slot_idx15_created,
        COUNT(*) AS jobs_in_sample,
        COUNT(DISTINCT day_local_created) AS sample_days,
        ROUND(
            COUNT(*)::numeric
            / NULLIF(COUNT(DISTINCT day_local_created), 0),
            2
        ) AS avg_jobs_per_day,
        ROUND(AVG(EXTRACT(EPOCH FROM duration)), 2) AS avg_duration_sec,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM duration)) AS p95_duration_sec,
        COUNT(*) FILTER (WHERE state = 'failed') AS failed_count,
        ROUND(100.0 * COUNT(*) FILTER (WHERE state = 'failed') / NULLIF(COUNT(*), 0), 2) AS failure_rate_pct
    FROM marie_scheduler.job
    WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 30
      AND slot_idx15_created IS NOT NULL
      AND duration IS NOT NULL
    GROUP BY slot_idx15_created
)
SELECT
    slot_idx15_created,
    LPAD((slot_idx15_created / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15_created % 4) * 15)::text, 2, '0') AS time_slot,
    jobs_in_sample,
    sample_days,
    avg_jobs_per_day,
    avg_duration_sec,
    p95_duration_sec,
    -- Worker equivalents for an average sampled day and 15-minute slot.
    CEIL(avg_jobs_per_day * avg_duration_sec / 900.0) AS recommended_workers,
    CEIL(avg_jobs_per_day * p95_duration_sec / 900.0) AS peak_capacity_workers,
    failure_rate_pct,
    CASE
        WHEN failure_rate_pct > 10 THEN 'HIGH_FAILURE'
        WHEN failure_rate_pct > 5 THEN 'MODERATE_FAILURE'
        ELSE 'HEALTHY'
    END AS health_status
FROM slot_metrics
ORDER BY slot_idx15_created;


-- 2.3 Compute-Demand Heatmap
-- Purpose: Estimate average worker-equivalent demand by creation slot
-- Output: Demand, not utilization; utilization requires provisioned capacity
-- =========================================================
SELECT
    slot_idx15_created,
    LPAD((slot_idx15_created / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15_created % 4) * 15)::text, 2, '0') AS time_slot,
    COUNT(*) AS total_jobs,
    SUM(EXTRACT(EPOCH FROM duration)) AS total_compute_seconds,
    ROUND(
        SUM(EXTRACT(EPOCH FROM duration))
        / NULLIF(COUNT(DISTINCT day_local_created), 0),
        2
    ) AS avg_compute_seconds_per_day,
    ROUND(
        SUM(EXTRACT(EPOCH FROM duration))
        / NULLIF(COUNT(DISTINCT day_local_created), 0)
        / 900.0,
        2
    ) AS avg_worker_equivalents,
    COUNT(DISTINCT day_local_created) AS sample_days
FROM marie_scheduler.job
WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 30
  AND slot_idx15_created IS NOT NULL
  AND duration IS NOT NULL
  AND state = 'completed'
GROUP BY slot_idx15_created
ORDER BY slot_idx15_created;


-- =========================================================
-- 3. SLA TRACKING: Compliance Analysis by Time Slot
-- =========================================================

-- 3.1 SLA Compliance Rate by Time Slot
-- Purpose: Track which time slots have highest SLA miss rates
-- Output: SLA compliance metrics per 15-min slot
-- =========================================================
WITH sla_events AS (
    SELECT
        'soft'::text AS sla_type,
        slot_idx15_soft AS slot_idx15,
        soft_sla AS deadline,
        completed_on
    FROM marie_scheduler.job
    WHERE day_local_soft >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 7
      AND soft_sla IS NOT NULL

    UNION ALL

    SELECT
        'hard',
        slot_idx15_hard,
        hard_sla,
        completed_on
    FROM marie_scheduler.job
    WHERE day_local_hard >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 7
      AND hard_sla IS NOT NULL
)
SELECT
    sla_type,
    slot_idx15,
    LPAD((slot_idx15 / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15 % 4) * 15)::text, 2, '0') AS sla_slot,
    COUNT(*) AS jobs_with_sla,
    COUNT(*) FILTER (
        WHERE completed_on IS NOT NULL AND completed_on <= deadline
    ) AS met_sla,
    COUNT(*) FILTER (
        WHERE completed_on IS NOT NULL AND completed_on > deadline
    ) AS missed_sla,
    COUNT(*) FILTER (
        WHERE completed_on IS NULL AND now() > deadline
    ) AS overdue_sla,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE completed_on IS NOT NULL AND completed_on <= deadline
        ) / NULLIF(COUNT(*) FILTER (
            WHERE completed_on IS NOT NULL
        ), 0),
        2
    ) AS sla_compliance_pct
FROM sla_events
WHERE slot_idx15 IS NOT NULL
GROUP BY sla_type, slot_idx15
ORDER BY sla_type, slot_idx15;


-- 3.2 Real-Time SLA Pressure Dashboard
-- Purpose: Show current SLA pressure for ongoing jobs
-- Output: Current state of SLA compliance in real-time
-- =========================================================
WITH current_sla_status AS (
    SELECT
        slot_idx15_effective,
        CASE
            WHEN hard_sla IS NOT NULL AND now() > hard_sla THEN 'HARD_MISSED'
            WHEN soft_sla IS NOT NULL AND now() > soft_sla THEN 'SOFT_MISSED'
            WHEN soft_sla IS NOT NULL AND now() > soft_sla - INTERVAL '15 minutes' THEN 'SOFT_WARNING'
            WHEN hard_sla IS NOT NULL AND now() > hard_sla - INTERVAL '15 minutes' THEN 'HARD_WARNING'
            ELSE 'OK'
        END AS sla_status,
        EXTRACT(EPOCH FROM (
            CASE
                WHEN hard_sla IS NOT NULL AND now() > hard_sla THEN hard_sla
                WHEN soft_sla IS NOT NULL THEN soft_sla
                ELSE hard_sla
            END - now()
        )) / 60 AS minutes_to_sla
    FROM marie_scheduler.job
    WHERE state IN ('created', 'retry', 'active')
      AND slot_idx15_effective IS NOT NULL
      AND (soft_sla IS NOT NULL OR hard_sla IS NOT NULL)
)
SELECT
    slot_idx15_effective,
    LPAD((slot_idx15_effective / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15_effective % 4) * 15)::text, 2, '0') AS time_slot,
    COUNT(*) AS total_jobs,
    COUNT(*) FILTER (WHERE sla_status = 'HARD_MISSED') AS hard_missed,
    COUNT(*) FILTER (WHERE sla_status = 'SOFT_MISSED') AS soft_missed,
    COUNT(*) FILTER (WHERE sla_status = 'HARD_WARNING') AS hard_warning,
    COUNT(*) FILTER (WHERE sla_status = 'SOFT_WARNING') AS soft_warning,
    COUNT(*) FILTER (WHERE sla_status = 'OK') AS on_track,
    ROUND(AVG(minutes_to_sla), 1) AS avg_minutes_to_deadline,
    MIN(minutes_to_sla) AS most_urgent_minutes_remaining
FROM current_sla_status
GROUP BY slot_idx15_effective
ORDER BY slot_idx15_effective;


-- 3.3 SLA Miss Reasons by Time Slot
-- Purpose: Understand why SLAs are missed in certain time slots
-- Output: Failure analysis by time slot
-- =========================================================
SELECT
    slot_idx15_soft,
    LPAD((slot_idx15_soft / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15_soft % 4) * 15)::text, 2, '0') AS time_slot,
    state,
    COUNT(*) AS job_count,
    COUNT(*) FILTER (WHERE completed_on > soft_sla) AS sla_misses,
    ROUND(AVG(
        CASE
            WHEN completed_on > soft_sla
            THEN EXTRACT(EPOCH FROM (completed_on - soft_sla))
        END
    ), 2) AS avg_overrun_seconds,
    ROUND(AVG(retry_count), 2) AS avg_retries,
    ROUND(100.0 * COUNT(*) FILTER (WHERE completed_on > soft_sla) / NULLIF(COUNT(*), 0), 2) AS miss_rate_pct
FROM marie_scheduler.job
WHERE day_local_soft >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 7
  AND soft_sla IS NOT NULL
  AND slot_idx15_soft IS NOT NULL
  AND completed_on IS NOT NULL
GROUP BY slot_idx15_soft, state
HAVING COUNT(*) FILTER (WHERE completed_on > soft_sla) > 0
ORDER BY slot_idx15_soft, sla_misses DESC;


-- =========================================================
-- 4. LOAD DISTRIBUTION: Peak Hours & Traffic Analysis
-- =========================================================

-- 4.1 Top 10 Peak Load Time Slots (Last 30 Days)
-- Purpose: Identify busiest periods for capacity planning
-- Output: Ranked list of peak time slots
-- =========================================================
SELECT
    slot_idx15_created,
    LPAD((slot_idx15_created / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15_created % 4) * 15)::text, 2, '0') AS time_slot,
    COUNT(*) AS total_jobs,
    COUNT(DISTINCT day_local_created) AS active_days,
    ROUND(COUNT(*) / NULLIF(COUNT(DISTINCT day_local_created), 0)::numeric, 2) AS avg_jobs_per_day,
    ROUND(AVG(priority), 2) AS avg_priority,
    COUNT(*) FILTER (WHERE state = 'failed') AS failed_jobs,
    ROUND(100.0 * COUNT(*) FILTER (WHERE state = 'failed') / NULLIF(COUNT(*), 0), 2) AS failure_rate_pct,
    RANK() OVER (ORDER BY COUNT(*) DESC) AS load_rank
FROM marie_scheduler.job
WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 30
  AND slot_idx15_created IS NOT NULL
GROUP BY slot_idx15_created
ORDER BY total_jobs DESC
LIMIT 10;


-- 4.2 Business Hours vs Off-Hours Analysis
-- Purpose: Compare workload during business hours vs off-hours
-- Output: Aggregated metrics for capacity planning
-- =========================================================
WITH categorized_slots AS (
    SELECT
        *,
        CASE
            WHEN (slot_idx15_created / 4) BETWEEN 9 AND 16 THEN 'Business Hours (9AM-5PM)'
            WHEN (slot_idx15_created / 4) BETWEEN 6 AND 8 THEN 'Early Morning (6AM-9AM)'
            WHEN (slot_idx15_created / 4) BETWEEN 17 AND 21 THEN 'Evening (5PM-10PM)'
            ELSE 'Night (10PM-6AM)'
        END AS time_category
    FROM marie_scheduler.job
    WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 30
      AND slot_idx15_created IS NOT NULL
)
SELECT
    time_category,
    COUNT(*) AS total_jobs,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
    COUNT(*) FILTER (WHERE state = 'completed') AS completed,
    COUNT(*) FILTER (WHERE state = 'failed') AS failed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE state = 'completed') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct,
    ROUND(AVG(EXTRACT(EPOCH FROM duration)), 2) AS avg_duration_sec,
    ROUND(AVG(priority), 2) AS avg_priority
FROM categorized_slots
GROUP BY time_category
ORDER BY total_jobs DESC;


-- 4.3 Daily Load Profile (24-Hour View)
-- Purpose: Show complete daily load profile
-- Output: Full 24-hour breakdown
-- =========================================================
WITH hourly_stats AS (
    SELECT
        (slot_idx15_created / 4) AS hour,
        COUNT(*) AS job_count,
        ROUND(AVG(EXTRACT(EPOCH FROM duration)), 2) AS avg_duration,
        COUNT(*) FILTER (WHERE state = 'failed') AS failures
    FROM marie_scheduler.job
    WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 7
      AND slot_idx15_created IS NOT NULL
    GROUP BY hour
)
SELECT
    hour,
    LPAD(hour::text, 2, '0') || ':00 - ' || LPAD((hour + 1)::text, 2, '0') || ':00' AS hour_range,
    job_count,
    avg_duration,
    failures,
    -- Show load intensity as visual bar
    REPEAT('█', LEAST(50, (job_count::numeric / MAX(job_count) OVER () * 50)::int)) AS load_bar,
    ROUND(100.0 * job_count / SUM(job_count) OVER (), 2) AS pct_of_daily_load
FROM hourly_stats
ORDER BY hour;


-- 4.4 Weekend vs Weekday Patterns
-- Purpose: Compare weekend and weekday load patterns
-- Output: Separate profiles for weekends and weekdays
-- =========================================================
SELECT
    CASE
        WHEN EXTRACT(DOW FROM day_local_created) IN (0, 6) THEN 'Weekend'
        ELSE 'Weekday'
    END AS day_type,
    (slot_idx15_created / 4) AS hour,
    LPAD((slot_idx15_created / 4)::text, 2, '0') || ':00' AS hour_label,
    COUNT(*) AS total_jobs,
    ROUND(AVG(priority), 2) AS avg_priority,
    COUNT(*) FILTER (WHERE state = 'completed') AS completed,
    COUNT(*) FILTER (WHERE state = 'failed') AS failed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE state = 'completed') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct
FROM marie_scheduler.job
WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 30
  AND slot_idx15_created IS NOT NULL
GROUP BY day_type, hour
ORDER BY day_type, hour;


-- =========================================================
-- 5. ADVANCED ANALYTICS: Combined Insights
-- =========================================================

-- 5.1 Current UTC-Day Effective-Slot Cohorts
-- Purpose: Summarize jobs whose effective SLA/creation day is today
-- Output: Current-state cohorts, not completion throughput
-- =========================================================
SELECT
    slot_idx15_effective,
    LPAD((slot_idx15_effective / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15_effective % 4) * 15)::text, 2, '0') AS time_slot,
    -- Volume metrics
    COUNT(*) AS total_jobs,
    COUNT(*) FILTER (WHERE state = 'completed') AS completed,
    COUNT(*) FILTER (WHERE state = 'active') AS active,
    COUNT(*) FILTER (WHERE state IN ('created', 'retry')) AS pending,
    COUNT(*) FILTER (WHERE state = 'failed') AS failed,
    -- Performance metrics
    ROUND(AVG(EXTRACT(EPOCH FROM duration)), 2) AS avg_duration_sec,
    ROUND(AVG(priority), 2) AS avg_priority,
    -- SLA metrics
    COUNT(*) FILTER (
        WHERE soft_sla IS NOT NULL
          AND now() > soft_sla
          AND state IN ('created', 'retry', 'active')
    ) AS soft_overdue,
    COUNT(*) FILTER (
        WHERE hard_sla IS NOT NULL
          AND now() > hard_sla
          AND state IN ('created', 'retry', 'active')
    ) AS hard_overdue,
    -- Success rate
    ROUND(100.0 * COUNT(*) FILTER (WHERE state = 'completed') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct
FROM marie_scheduler.job
WHERE day_local_effective = marie_scheduler.day_in_tz(NOW(), 'UTC')
  AND slot_idx15_effective IS NOT NULL
GROUP BY slot_idx15_effective
ORDER BY slot_idx15_effective;


-- 5.2 Trend Analysis: Week-over-Week Comparison
-- Purpose: Compare current week to previous week
-- Output: Trend indicators for each metric
-- =========================================================
WITH current_week AS (
    SELECT
        slot_idx15_created,
        COUNT(*) AS jobs,
        COUNT(*) FILTER (WHERE state = 'failed') AS failures
    FROM marie_scheduler.job
    WHERE day_local_created >= DATE_TRUNC(
        'week',
        marie_scheduler.day_in_tz(NOW(), 'UTC')::timestamp
    )::date
      AND slot_idx15_created IS NOT NULL
    GROUP BY slot_idx15_created
),
previous_week AS (
    SELECT
        slot_idx15_created,
        COUNT(*) AS jobs,
        COUNT(*) FILTER (WHERE state = 'failed') AS failures
    FROM marie_scheduler.job
    WHERE day_local_created >= DATE_TRUNC(
        'week',
        marie_scheduler.day_in_tz(NOW(), 'UTC')::timestamp
    )::date - 7
      AND day_local_created < DATE_TRUNC(
          'week',
          marie_scheduler.day_in_tz(NOW(), 'UTC')::timestamp
      )::date
      AND slot_idx15_created IS NOT NULL
    GROUP BY slot_idx15_created
)
SELECT
    c.slot_idx15_created,
    LPAD((c.slot_idx15_created / 4)::text, 2, '0') || ':' ||
    LPAD(((c.slot_idx15_created % 4) * 15)::text, 2, '0') AS time_slot,
    c.jobs AS current_week_jobs,
    COALESCE(p.jobs, 0) AS previous_week_jobs,
    c.jobs - COALESCE(p.jobs, 0) AS jobs_change,
    ROUND(100.0 * (c.jobs - COALESCE(p.jobs, 0)) / NULLIF(p.jobs, 0), 2) AS jobs_change_pct,
    c.failures AS current_week_failures,
    COALESCE(p.failures, 0) AS previous_week_failures,
    CASE
        WHEN c.jobs > COALESCE(p.jobs, 0) * 1.2 THEN '📈 INCREASING'
        WHEN c.jobs < COALESCE(p.jobs, 0) * 0.8 THEN '📉 DECREASING'
        ELSE '➡️  STABLE'
    END AS trend
FROM current_week c
LEFT JOIN previous_week p ON c.slot_idx15_created = p.slot_idx15_created
ORDER BY c.slot_idx15_created;


-- =========================================================
-- 6. EXPORT FORMATS: Visualization-Ready Outputs
-- =========================================================

-- 6.1 JSON Export for Heatmap Visualization Tools
-- Purpose: Export data in JSON format for charting libraries (Chart.js, D3.js, etc.)
-- Output: JSON array of time slot data
-- =========================================================
SELECT json_agg(
    json_build_object(
        'slot', slot_idx15_created,
        'hour', (slot_idx15_created / 4),
        'minute', ((slot_idx15_created % 4) * 15),
        'label', LPAD((slot_idx15_created / 4)::text, 2, '0') || ':' ||
                 LPAD(((slot_idx15_created % 4) * 15)::text, 2, '0'),
        'jobs', job_count,
        'completed', completed_count,
        'failed', failed_count,
        'success_rate', success_rate
    )
    ORDER BY slot_idx15_created
) AS heatmap_data
FROM (
    SELECT
        slot_idx15_created,
        COUNT(*) AS job_count,
        COUNT(*) FILTER (WHERE state = 'completed') AS completed_count,
        COUNT(*) FILTER (WHERE state = 'failed') AS failed_count,
        ROUND(100.0 * COUNT(*) FILTER (WHERE state = 'completed') / NULLIF(COUNT(*), 0), 2) AS success_rate
    FROM marie_scheduler.job
    WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 7
      AND slot_idx15_created IS NOT NULL
    GROUP BY slot_idx15_created
) subquery;


-- 6.2 CSV-Ready Format for Excel/Spreadsheet Analysis
-- Purpose: Simple format for export to CSV
-- Output: Flat table ready for CSV export
-- =========================================================
SELECT
    slot_idx15_created AS slot,
    LPAD((slot_idx15_created / 4)::text, 2, '0') || ':' ||
    LPAD(((slot_idx15_created % 4) * 15)::text, 2, '0') AS time,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE state = 'completed') AS completed,
    COUNT(*) FILTER (WHERE state = 'active') AS active,
    COUNT(*) FILTER (WHERE state = 'failed') AS failed,
    ROUND(AVG(priority), 2) AS avg_priority,
    ROUND(AVG(EXTRACT(EPOCH FROM duration)), 2) AS avg_duration_sec
FROM marie_scheduler.job
WHERE day_local_created >= marie_scheduler.day_in_tz(NOW(), 'UTC') - 7
  AND slot_idx15_created IS NOT NULL
GROUP BY slot_idx15_created
ORDER BY slot_idx15_created;


-- =========================================================
-- NOTES:
-- =========================================================
-- 1. All queries use generated columns (slot_idx15_*, day_local_*)
--    which are precomputed for performance.
--
-- 2. Timezone: All data is stored in UTC.
--    - Despite "local" in column names, all calculations use UTC timezone
--    - All day_local_* columns contain UTC calendar dates
--    - The "local" name indicates conceptual purpose, not stored timezone
--    - For display in specific timezone (e.g., America/Chicago):
--      SELECT created_on AT TIME ZONE 'America/Chicago' AS chicago_time
--    - For filtering by specific timezone date:
--      WHERE (created_on AT TIME ZONE 'America/Chicago')::date = '2025-01-04'
--
-- 3. Date ranges can be adjusted via WHERE clauses:
--    - day_in_tz(NOW(), 'UTC') - 7   (last week in UTC)
--    - day_in_tz(NOW(), 'UTC') - 30  (last month in UTC)
--    - day_in_tz(NOW(), 'UTC')       (today in UTC)
--
-- 4. Performance: Queries on day_local_effective can use the composite
--    (day_local_effective, slot_idx15_effective) index. Other day/slot
--    combinations require plan review and may scan queue partitions.
--
-- 5. Visualization: JSON and CSV export formats provided for
--    integration with charting libraries and BI tools.
--
-- 6. Timezone Conversion Examples:
--    -- Convert slot back to specific timezone for display
--    SELECT to_char(
--        (day_local_created + (slot_idx15_created * interval '15 minutes'))
--          AT TIME ZONE 'UTC'
--          AT TIME ZONE 'America/Chicago',
--        'YYYY-MM-DD HH24:MI TZ'
--    ) AS chicago_time;
--
--    -- Group by specific timezone hour
--    SELECT
--        EXTRACT(HOUR FROM created_on AT TIME ZONE 'America/Chicago') AS chicago_hour,
--        COUNT(*)
--    FROM marie_scheduler.job
--    GROUP BY chicago_hour;
-- =========================================================
