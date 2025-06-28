
CREATE OR REPLACE VIEW marie_scheduler.ready_jobs_view AS
WITH candidate_jobs AS (
    SELECT *
    FROM marie_scheduler.job
    WHERE state < 'active'
      AND start_after < now()
),
unblocked_job_ids AS (
    SELECT j.name, j.id
    FROM candidate_jobs j
    LEFT JOIN marie_scheduler.job_dependencies dep
      ON dep.job_name = j.name AND dep.job_id = j.id
    LEFT JOIN marie_scheduler.job d
      ON d.name = dep.depends_on_name AND d.id = dep.depends_on_id
    GROUP BY j.name, j.id
    HAVING
      COUNT(dep.depends_on_id) = 0 OR
      COUNT(*) FILTER (WHERE d.state IS DISTINCT FROM 'completed') = 0
)
SELECT j.*
FROM candidate_jobs j
JOIN unblocked_job_ids u
  ON j.name = u.name AND j.id = u.id
WHERE NOT EXISTS (
    SELECT 1
    FROM marie_scheduler.dag d
    WHERE d.id = j.dag_id AND d.state = 'completed'
);