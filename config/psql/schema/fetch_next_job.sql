CREATE OR REPLACE FUNCTION marie_scheduler.fetch_next_job(
    job_name   text,
    batch_size integer DEFAULT 250,
    w_hot      numeric DEFAULT 0.70
)
RETURNS SETOF marie_scheduler.job
LANGUAGE sql
STABLE
PARALLEL SAFE
SET work_mem = '256MB'
SET max_parallel_workers_per_gather = '4'
SET enable_partitionwise_join = 'on'
AS $$
WITH
eligible AS MATERIALIZED (
  SELECT
    j.id, j.dag_id, j.job_level, j.priority, j.created_on,
    d.state AS dag_state
  FROM marie_scheduler.job j
  JOIN marie_scheduler.dag d ON d.id = j.dag_id
  WHERE d.state IN ('active','created')
    AND j.name  = fetch_next_job.job_name
    AND j.state IN ('created','retry')
),

candidate AS MATERIALIZED (
  SELECT e.*
  FROM eligible e
  WHERE NOT EXISTS (
    SELECT 1
    FROM marie_scheduler.job_dependencies dep
    LEFT JOIN marie_scheduler.job p
      ON p.id = dep.depends_on_id
      AND p.dag_id = e.dag_id
    WHERE dep.job_id = e.id
      AND (
        p.id IS NULL
        OR p.state <> 'completed'
      )
  )
),

ready AS (
  SELECT
    COUNT(*) FILTER (WHERE dag_state = 'active')  AS ready_hot,
    COUNT(*) FILTER (WHERE dag_state = 'created') AS ready_new
  FROM candidate
),
caps AS (
  SELECT
    fetch_next_job.batch_size                                   AS batch_sz,
    CEIL(fetch_next_job.batch_size * fetch_next_job.w_hot)::int AS hot_cap,
    r.ready_hot, r.ready_new
  FROM ready r
),
alloc AS (
  SELECT
    batch_sz,
    hot_cap,
    ready_hot, ready_new,
    LEAST(hot_cap, ready_hot)                              AS hot_take,
    LEAST(batch_sz - LEAST(hot_cap, ready_hot), ready_new) AS new_take,
    GREATEST(
      batch_sz
      - LEAST(hot_cap, ready_hot)
      - LEAST(batch_sz - LEAST(hot_cap, ready_hot), ready_new),
      0
    ) AS remaining
  FROM caps
),
final_caps AS (
  SELECT
    batch_sz,
    CASE
      WHEN remaining = 0 THEN hot_take
      WHEN (ready_hot - hot_take) >= (ready_new - new_take)
        THEN hot_take + LEAST(remaining, GREATEST(ready_hot - hot_take, 0))
      ELSE hot_take
    END AS hot_total,
    CASE
      WHEN remaining = 0 THEN new_take
      WHEN (ready_hot - hot_take) < (ready_new - new_take)
        THEN new_take + LEAST(remaining, GREATEST(ready_new - new_take, 0))
      ELSE new_take
    END AS new_total
  FROM alloc
),

pick_hot AS MATERIALIZED (
  SELECT id, dag_id, job_level, priority, dag_state
  FROM candidate
  WHERE dag_state = 'active'
  ORDER BY job_level DESC, priority DESC, id
  LIMIT (SELECT hot_total FROM final_caps)
),
pick_new AS MATERIALIZED (
  SELECT id, dag_id, job_level, priority, dag_state
  FROM candidate
  WHERE dag_state = 'created'
  ORDER BY job_level DESC, priority DESC, id
  LIMIT (SELECT new_total FROM final_caps)
),

picked AS MATERIALIZED (
  SELECT * FROM pick_hot
  UNION ALL
  SELECT * FROM pick_new
  LIMIT fetch_next_job.batch_size
)

SELECT j.*
FROM picked p
JOIN marie_scheduler.job j ON j.id = p.id
ORDER BY p.job_level DESC, p.priority DESC, j.id;
$$;

ALTER FUNCTION fetch_next_job(text, integer, numeric) OWNER TO postgres;
