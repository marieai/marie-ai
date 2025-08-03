DROP VIEW IF EXISTS marie_scheduler.dag_bucketed_sla_view CASCADE;

CREATE OR REPLACE VIEW marie_scheduler.dag_bucketed_sla_view AS
WITH job_with_sla AS (
    SELECT
        j.*,
        d.name AS dag_name,
        NOW() AS now_ts
    FROM marie_scheduler.job j
    JOIN marie_scheduler.dag d ON d.id = j.dag_id
    WHERE j.soft_sla IS NOT NULL OR j.hard_sla IS NOT NULL
),
bucketed_jobs AS (
    SELECT
        name AS job_name,
        dag_id,
        dag_name,
        state,
        priority,
        now_ts,
        hard_sla,
        soft_sla,

        -- Determine SLA type and bucket index
        CASE
            WHEN hard_sla IS NOT NULL AND now_ts > hard_sla THEN 'hard'
            WHEN soft_sla IS NOT NULL AND now_ts > soft_sla THEN 'soft'
            ELSE NULL
        END AS overdue_type,

        CASE
            WHEN hard_sla IS NOT NULL AND now_ts > hard_sla THEN
                FLOOR(EXTRACT(EPOCH FROM (now_ts - hard_sla)) / 900)::int
            WHEN soft_sla IS NOT NULL AND now_ts > soft_sla THEN
                FLOOR(EXTRACT(EPOCH FROM (now_ts - soft_sla)) / 900)::int
            ELSE NULL
        END AS overdue_buckets,

        CASE
            WHEN soft_sla IS NOT NULL AND now_ts <= soft_sla THEN
                FLOOR(EXTRACT(EPOCH FROM (soft_sla - now_ts)) / 900)::int
            ELSE NULL
        END AS future_bucket_index_soft,

        CASE
            WHEN hard_sla IS NOT NULL AND now_ts <= hard_sla THEN
                FLOOR(EXTRACT(EPOCH FROM (hard_sla - now_ts)) / 900)::int
            ELSE NULL
        END AS future_bucket_index_hard,

        -- Unified index for ordering
        COALESCE(
            FLOOR(EXTRACT(EPOCH FROM (hard_sla - now_ts)) / 900),
            FLOOR(EXTRACT(EPOCH FROM (soft_sla - now_ts)) / 900),
            -1 * FLOOR(EXTRACT(EPOCH FROM (now_ts - COALESCE(hard_sla, soft_sla))) / 900)
        ) AS bucket_index
    FROM job_with_sla
)
SELECT
    job_name,
    dag_id,
    overdue_type,
    overdue_buckets,
    bucket_index,

    -- Human-friendly label
    CASE
        WHEN overdue_type IS NOT NULL THEN
            CONCAT(
                'overdue ',
                overdue_buckets * 15,
                '–',
                (overdue_buckets + 1) * 15,
                ' min (',
                overdue_type,
                ')'
            )
        WHEN future_bucket_index_hard IS NOT NULL THEN
            CONCAT(
                'HARD SLA due in ',
                future_bucket_index_hard * 15,
                '–',
                (future_bucket_index_hard + 1) * 15,
                ' min'
            )
        WHEN future_bucket_index_soft IS NOT NULL THEN
            CONCAT(
                'SOFT SLA due in ',
                future_bucket_index_soft * 15,
                '–',
                (future_bucket_index_soft + 1) * 15,
                ' min'
            )
        ELSE 'no SLA'
    END AS bucket_label,

    COUNT(*) AS total_jobs,
    COUNT(*) FILTER (WHERE state = 'completed') AS completed_jobs,
    COUNT(*) FILTER (WHERE state <> 'completed') AS outstanding_jobs,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE state = 'completed') / NULLIF(COUNT(*), 0),
        1
    ) AS percent_complete

FROM bucketed_jobs
WHERE
    overdue_type IS NOT NULL
    OR future_bucket_index_soft IS NOT NULL
    OR future_bucket_index_hard IS NOT NULL
GROUP BY
    job_name,
    dag_id,
    overdue_type,
    overdue_buckets,
    future_bucket_index_soft,
    future_bucket_index_hard,
    bucket_index
ORDER BY
    bucket_index,
    dag_id,
    job_name;
