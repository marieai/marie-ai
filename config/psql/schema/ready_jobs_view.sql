CREATE OR REPLACE VIEW marie_scheduler.ready_jobs_view AS
WITH candidate_jobs AS (
    SELECT *,
       CASE
          WHEN NOW() > hard_sla THEN 'missed_hard'
          WHEN NOW() > soft_sla THEN 'missed_soft'
          WHEN soft_sla IS NOT NULL THEN 'on_time'
          ELSE 'no_sla'
        END AS sla_tier
    FROM marie_scheduler.job
    WHERE state < 'active'
      AND start_after < now()
),
unblocked_jobs AS (
    SELECT j.*
    FROM candidate_jobs j
    LEFT JOIN LATERAL (
        SELECT 1
        FROM jsonb_array_elements_text(j.dependencies) dep(val)
        JOIN marie_scheduler.job d ON d.id = dep.val::uuid
        WHERE d.state != 'completed'
        LIMIT 1
    ) blocked ON TRUE
    WHERE blocked IS NULL
)
SELECT j.*
FROM unblocked_jobs j
WHERE NOT EXISTS (
    SELECT 1
    FROM marie_scheduler.dag d
    WHERE d.id = j.dag_id
      AND d.state = 'completed'
)

