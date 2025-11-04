DROP VIEW IF EXISTS marie_scheduler.dag_bucketed_sla_view CASCADE;

CREATE OR REPLACE VIEW marie_scheduler.dag_bucketed_sla_view AS
WITH job_with_sla AS (
  SELECT
    j.*,
    d.name AS dag_name,
    NOW() AT TIME ZONE 'America/Chicago' AS now_ts,
    marie_scheduler.local_day(NOW() AT TIME ZONE 'America/Chicago', 'America/Chicago') AS now_day_local,
    marie_scheduler.slot_15m(NOW() AT TIME ZONE 'America/Chicago', 'America/Chicago')  AS now_slot_idx
  FROM marie_scheduler.job j
  JOIN marie_scheduler.dag d ON d.id = j.dag_id
  WHERE j.soft_sla IS NOT NULL OR j.hard_sla IS NOT NULL
),
slot_deltas AS (
  SELECT
    j.name AS job_name,
    j.dag_id,
    j.dag_name,
    j.state,
    j.priority,
    j.now_ts,
    j.now_day_local,
    j.now_slot_idx,
    j.soft_sla,
    j.hard_sla,
    j.day_local_soft,
    j.slot_idx15_soft,
    j.day_local_hard,
    j.slot_idx15_hard,

    /* compute the soft/hard slot deltas */
    CASE
      WHEN j.day_local_soft IS NOT NULL THEN
        ((j.day_local_soft - j.now_day_local) * 96)::int + (j.slot_idx15_soft - j.now_slot_idx)
      ELSE NULL
    END AS soft_bucket_index,

    CASE
      WHEN j.day_local_hard IS NOT NULL THEN
        ((j.day_local_hard - j.now_day_local) * 96)::int + (j.slot_idx15_hard - j.now_slot_idx)
      ELSE NULL
    END AS hard_bucket_index
  FROM job_with_sla j
),
bucketed_jobs AS (
  SELECT
    *,
    CASE
      WHEN hard_bucket_index IS NOT NULL AND hard_bucket_index < 0 THEN 'hard'
      WHEN soft_bucket_index IS NOT NULL AND soft_bucket_index < 0 THEN 'soft'
      ELSE NULL
    END AS overdue_type,

    CASE
      WHEN hard_bucket_index IS NOT NULL AND hard_bucket_index < 0 THEN ABS(hard_bucket_index)
      WHEN soft_bucket_index IS NOT NULL AND soft_bucket_index < 0 THEN ABS(soft_bucket_index)
      ELSE NULL
    END AS overdue_buckets,

    CASE
      WHEN soft_bucket_index IS NOT NULL AND soft_bucket_index >= 0 THEN soft_bucket_index
      ELSE NULL
    END AS future_bucket_index_soft,

    CASE
      WHEN hard_bucket_index IS NOT NULL AND hard_bucket_index >= 0 THEN hard_bucket_index
      ELSE NULL
    END AS future_bucket_index_hard,

    COALESCE(hard_bucket_index, soft_bucket_index) AS bucket_index
  FROM slot_deltas
)
SELECT
  job_name,
  dag_id,
  dag_name,
  overdue_type,
  overdue_buckets,
  bucket_index,

  CASE
    WHEN overdue_type IS NOT NULL THEN
      CONCAT('overdue ', overdue_buckets * 15, '–', (overdue_buckets + 1) * 15, ' min (', overdue_type, ')')
    WHEN future_bucket_index_hard IS NOT NULL THEN
      CONCAT('HARD SLA due in ', future_bucket_index_hard * 15, '–', (future_bucket_index_hard + 1) * 15, ' min')
    WHEN future_bucket_index_soft IS NOT NULL THEN
      CONCAT('SOFT SLA due in ', future_bucket_index_soft * 15, '–', (future_bucket_index_soft + 1) * 15, ' min')
    ELSE 'no SLA'
  END AS bucket_label,

  COUNT(*) AS total_jobs,
  COUNT(*) FILTER (WHERE state = 'completed') AS completed_jobs,
  COUNT(*) FILTER (WHERE state <> 'completed') AS outstanding_jobs,
  ROUND(
    100.0 * COUNT(*) FILTER (WHERE state = 'completed') / NULLIF(COUNT(*), 0), 1
  ) AS percent_complete

FROM bucketed_jobs
WHERE
  overdue_type IS NOT NULL
  OR future_bucket_index_soft IS NOT NULL
  OR future_bucket_index_hard IS NOT NULL
GROUP BY
  job_name,
  dag_id,
  dag_name,
  overdue_type,
  overdue_buckets,
  future_bucket_index_soft,
  future_bucket_index_hard,
  bucket_index
ORDER BY
  bucket_index,
  dag_id,
  job_name;

