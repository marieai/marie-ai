CREATE OR REPLACE FUNCTION marie_scheduler.fetch_next_job(
    job_name             text,
    batch_size           integer DEFAULT 5000,
    max_concurrent_dags  integer DEFAULT NULL  -- NULL => no gating
)
    RETURNS SETOF marie_scheduler.job
    LANGUAGE sql
    STABLE
    PARALLEL SAFE
    SET work_mem = '256MB'
    SET max_parallel_workers_per_gather = '4'
    SET enable_partitionwise_join = 'on'
    AS $$
    WITH active_dags AS MATERIALIZED (
      SELECT id
      FROM marie_scheduler.dag
      WHERE state NOT IN ('completed','failed','cancelled')
    ),
    running_dags AS MATERIALIZED (
      SELECT DISTINCT j.dag_id
      FROM marie_scheduler.job j
      WHERE j.state = 'active'
        AND j.dag_id IN (SELECT id FROM active_dags)
    ),
    -- Count how many DAGs are currently "running" (have active jobs)
    running_count AS (
      SELECT COUNT(*)::int AS n FROM running_dags
    ),
    -- How many non-running active DAGs exist?
    non_running_active AS (
      SELECT COUNT(*)::int AS m
      FROM active_dags a
      WHERE a.id NOT IN (SELECT dag_id FROM running_dags)
    ),
    -- Compute slots if gating is enabled; if max_concurrent_dags is NULL, treat as "infinite"
    allowed_new AS (
      SELECT CASE
               WHEN fetch_next_job.max_concurrent_dags IS NULL THEN
                 -- effectively allow all remaining active DAGs
                 (SELECT m FROM non_running_active)
               ELSE
                 GREATEST(fetch_next_job.max_concurrent_dags - (SELECT n FROM running_count), 0)
             END::int AS slots
    ),
    /* Ready = created/retry AND no open deps, only on active DAGs */
    ready AS MATERIALIZED (
      SELECT j.id, j.dag_id, j.job_level, j.priority
      FROM marie_scheduler.job j
      WHERE j.name = fetch_next_job.job_name
        AND j.state IN ('created','retry')
        AND j.dag_id IN (SELECT id FROM active_dags)
        AND NOT EXISTS (
          SELECT 1
          FROM marie_scheduler.job_dependencies dep
          JOIN marie_scheduler.job dj ON dj.id = dep.depends_on_id
          WHERE dep.job_id = j.id
            AND dj.state <> 'completed'
        )
    ),
    /* Admit up to `slots` NEW DAGs by best ready root (job_level = 0) */
    new_dag_ids AS MATERIALIZED (
      SELECT r.dag_id
      FROM ready r
      WHERE r.job_level = 0
        AND r.dag_id NOT IN (SELECT dag_id FROM running_dags)
      GROUP BY r.dag_id
      ORDER BY MAX(r.priority) DESC
      LIMIT (SELECT slots FROM allowed_new)
    ),
    /* All ready jobs from already-running DAGs + all ready jobs from newly admitted DAGs */
    pick AS (
      SELECT id, dag_id, job_level, priority
      FROM ready
      WHERE dag_id IN (SELECT dag_id FROM running_dags)
      UNION ALL
      SELECT id, dag_id, job_level, priority
      FROM ready
      WHERE dag_id IN (SELECT dag_id FROM new_dag_ids)
    )
    SELECT j.*
    FROM pick p
    JOIN marie_scheduler.job j ON j.id = p.id
    ORDER BY
      p.job_level  ASC,
      p.priority   DESC
    LIMIT fetch_next_job.batch_size;
$$;
