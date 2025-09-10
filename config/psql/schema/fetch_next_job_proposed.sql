CREATE OR REPLACE FUNCTION marie_scheduler.fetch_next_job(
    job_name   text,
    batch_size integer DEFAULT 250,
    w_running  numeric DEFAULT 0.70,
    overfetch  integer DEFAULT 6
)
RETURNS SETOF marie_scheduler.job
LANGUAGE sql
STABLE
PARALLEL SAFE
SET work_mem = '256MB'
SET max_parallel_workers_per_gather = '4'
SET enable_partitionwise_join = 'on'
AS $$
WITH caps AS (
  SELECT
    GREATEST(1, batch_size)::int AS batch_sz,
    CEIL(GREATEST(1, batch_size) * w_running)::int AS cap_running
),
active_dags AS MATERIALIZED (
  SELECT id FROM marie_scheduler.dag WHERE state = 'active'
),
new_dags AS MATERIALIZED (
  SELECT id FROM marie_scheduler.dag WHERE state = 'created'
),
-- 1) PRE-SELECTION (split lanes before LIMIT)
ordered_running_jobs AS MATERIALIZED (
  SELECT j.id, j.dag_id, j.job_level, j.priority
  FROM marie_scheduler.job j
  JOIN active_dags ad ON ad.id = j.dag_id
  WHERE j.name = job_name
    AND j.state IN ('created','retry')
  ORDER BY j.job_level ASC, j.priority DESC
  LIMIT GREATEST(1000, (SELECT cap_running FROM caps) * GREATEST(1, overfetch))
),
ordered_new_jobs AS MATERIALIZED (
  SELECT j.id, j.dag_id, j.job_level, j.priority
  FROM marie_scheduler.job j
  JOIN new_dags nd ON nd.id = j.dag_id
  WHERE j.name = job_name
    AND j.state IN ('created','retry')
  ORDER BY j.job_level ASC, j.priority DESC
  LIMIT GREATEST(1000,
         ((SELECT batch_sz FROM caps) - (SELECT cap_running FROM caps))
         * GREATEST(1, overfetch))
),
ordered_jobs AS MATERIALIZED (
  SELECT * FROM ordered_running_jobs
  UNION ALL
  SELECT * FROM ordered_new_jobs
),
-- 2) Dependency prune
dep_for_active AS (
  SELECT dep.job_id, dep.depends_on_id
  FROM marie_scheduler.job_dependencies dep
  JOIN ordered_jobs oj ON oj.id = dep.job_id
),
open_deps_subset AS (
  SELECT dfa.job_id
  FROM dep_for_active dfa
  JOIN marie_scheduler.job dj ON dj.id = dfa.depends_on_id
  WHERE dj.state <> 'completed'
  GROUP BY dfa.job_id
),
candidate AS MATERIALIZED (
  SELECT oj.id, oj.dag_id, oj.job_level, oj.priority
  FROM ordered_jobs oj
  LEFT JOIN open_deps_subset od ON od.job_id = oj.id
  WHERE od.job_id IS NULL
),
-- 3) Final pick with class caps
pick_running AS MATERIALIZED (
  SELECT *
  FROM candidate c
  WHERE EXISTS (SELECT 1 FROM active_dags ad WHERE ad.id = c.dag_id)
  ORDER BY job_level ASC, priority DESC
  LIMIT (SELECT cap_running FROM caps)
),
pick_new AS MATERIALIZED (
  SELECT *
  FROM candidate c
  WHERE EXISTS (SELECT 1 FROM new_dags nd WHERE nd.id = c.dag_id)
  ORDER BY job_level ASC, priority DESC
  LIMIT GREATEST(0, (SELECT batch_sz FROM caps) - (SELECT COUNT(*) FROM pick_running))
),
chosen AS (
  SELECT id FROM pick_running
  UNION ALL
  SELECT id FROM pick_new
)
SELECT j.*
FROM chosen c
JOIN marie_scheduler.job j ON j.id = c.id;
$$;
