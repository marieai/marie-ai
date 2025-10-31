-- :target_day :: date (America/Chicago local day), e.g. '2025-10-31'

WITH live AS (
  SELECT *
  FROM marie_scheduler.job
  WHERE state IN ('queued','running')
),
by_basis AS (
  -- SOFT
  SELECT 'soft'::text AS basis, day_local_soft AS day_local, slot_idx15_soft AS slot_index, priority
  FROM live
  WHERE day_local_soft = :target_day

  UNION ALL
  -- HARD
  SELECT 'hard', day_local_hard, slot_idx15_hard, priority
  FROM live
  WHERE day_local_hard = :target_day

  UNION ALL
  -- EFFECTIVE (soft -> hard -> created_on)
  SELECT 'effective', day_local_effective, slot_idx15_effective, priority
  FROM live
  WHERE day_local_effective = :target_day

  UNION ALL
  -- CREATED_ON
  SELECT 'created', day_local_created, slot_idx15_created, priority
  FROM live
  WHERE day_local_created = :target_day
),
counts AS (
  SELECT
    basis,
    slot_index,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE priority >= 1000)            AS hard_breach,
    COUNT(*) FILTER (WHERE priority BETWEEN 500 AND 999) AS soft_breach,
    COUNT(*) FILTER (WHERE priority BETWEEN 1 AND 499)   AS approaching,
    COUNT(*) FILTER (WHERE priority <= 0)                AS low_or_legacy
  FROM by_basis
  GROUP BY basis, slot_index
),
slots AS (SELECT generate_series(0,95) AS slot_index),
bases AS (SELECT unnest(ARRAY['soft','hard','effective','created']::text[]) AS basis)
SELECT
  b.basis,
  :target_day::date AS day_local,
  json_agg(
    json_build_object(
      'slot', s.slot_index,
      'total', COALESCE(c.total, 0),
      'hard',  COALESCE(c.hard_breach, 0),
      'soft',  COALESCE(c.soft_breach, 0),
      'near',  COALESCE(c.approaching, 0),
      'low',   COALESCE(c.low_or_legacy, 0)
    )
    ORDER BY s.slot_index
  ) AS slots_15m
FROM bases b
CROSS JOIN slots s
LEFT JOIN counts c
  ON c.basis = b.basis AND c.slot_index = s.slot_index
GROUP BY b.basis
ORDER BY b.basis;


CREATE INDEX IF NOT EXISTS job_state_day_soft_idx      ON marie_scheduler.job (state, day_local_soft);
CREATE INDEX IF NOT EXISTS job_state_day_hard_idx      ON marie_scheduler.job (state, day_local_hard);
CREATE INDEX IF NOT EXISTS job_state_day_effective_idx ON marie_scheduler.job (state, day_local_effective);
CREATE INDEX IF NOT EXISTS job_state_day_created_idx   ON marie_scheduler.job (state, day_local_created);
