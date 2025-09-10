CREATE OR REPLACE FUNCTION marie_scheduler.fetch_next_job(
    job_name   text,
    batch_size integer DEFAULT 1,
    w_running  numeric DEFAULT 0.70        -- fraction of batch reserved for running DAGs
)
RETURNS SETOF marie_scheduler.job
LANGUAGE sql
STABLE
PARALLEL SAFE
SET work_mem = '256MB'
SET max_parallel_workers_per_gather = '4'
SET enable_partitionwise_join = 'on'
AS $$
WITH ordered_jobs AS MATERIALIZED (
  SELECT id, dag_id, job_level, priority
  FROM marie_scheduler.job
  WHERE name = job_name
    AND state IN ('created','retry')
  ORDER BY job_level ASC, priority DESC
  LIMIT 12000
),
active_jobs AS MATERIALIZED (
  SELECT oj.*
  FROM ordered_jobs oj
  WHERE EXISTS (
    SELECT 1
    FROM marie_scheduler.dag d
    WHERE d.id = oj.dag_id
      AND d.state NOT IN ('completed','failed','cancelled')
  )
),
dep_for_active AS (
  SELECT dep.job_id, dep.depends_on_id
  FROM marie_scheduler.job_dependencies dep
  JOIN active_jobs aj ON aj.id = dep.job_id
),
open_deps_subset AS (
  SELECT dfa.job_id
  FROM dep_for_active dfa
  JOIN marie_scheduler.job dj ON dj.id = dfa.depends_on_id
  WHERE dj.state <> 'completed'
  GROUP BY dfa.job_id
),
-- Ready = dependency-free and still active
candidate AS MATERIALIZED (
  SELECT aj.id, aj.dag_id, aj.job_level, aj.priority
  FROM active_jobs aj
  LEFT JOIN open_deps_subset od ON od.job_id = aj.id
  WHERE od.job_id IS NULL
  LIMIT 5000
),
-- DAGs that already have work "in flight"
running_dags AS MATERIALIZED (
  SELECT DISTINCT j.dag_id
  FROM marie_scheduler.job j
  WHERE j.state = 'active'
),
caps AS (
  SELECT
    GREATEST(1, batch_size)::int                    AS batch_sz,
    CEIL(GREATEST(1, batch_size) * w_running)::int  AS cap_running
),
pick_running AS MATERIALIZED (
  SELECT *
  FROM candidate
  WHERE dag_id IN (SELECT dag_id FROM running_dags)
  ORDER BY job_level ASC, priority DESC
  LIMIT (SELECT cap_running FROM caps)
),
pick_new AS MATERIALIZED (
  SELECT *
  FROM candidate
  WHERE dag_id NOT IN (SELECT dag_id FROM running_dags)
  ORDER BY job_level ASC, priority DESC
  LIMIT GREATEST(
          0,
          (SELECT batch_sz FROM caps) - (SELECT COUNT(*) FROM pick_running)
        )
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
