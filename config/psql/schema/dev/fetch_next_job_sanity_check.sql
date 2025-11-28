-- This script is used to sanity check the fetch_next_job function in the scheduler.

-- 1) Do we actually have candidates?
WITH candidate_jobs AS (
  SELECT j.*
  FROM marie_scheduler.job j
  JOIN marie_scheduler.dag d ON d.id = j.dag_id
  WHERE j.name = 'gen5_extract'
    AND j.state < 'active'
    AND j.start_after < now()
    AND d.state != 'completed'
)
SELECT count(*) FROM candidate_jobs;

-- 2) Which of those are blocked by unmet deps?
WITH candidate_jobs AS (
  SELECT j.*
  FROM marie_scheduler.job j
  JOIN marie_scheduler.dag d ON d.id = j.dag_id
  WHERE j.name = 'gen5_extract'
    AND j.state < 'active'
    AND j.start_after < now()
    AND d.state != 'completed'
)
SELECT j.id, j.name
FROM candidate_jobs j
WHERE EXISTS (
  SELECT 1
  FROM marie_scheduler.job_dependencies dep
  JOIN marie_scheduler.job d2 ON d2.name = dep.depends_on_name AND d2.id = dep.depends_on_id
  WHERE dep.job_name = j.name
    AND dep.job_id   = j.id
    AND d2.state <> 'completed'
);
