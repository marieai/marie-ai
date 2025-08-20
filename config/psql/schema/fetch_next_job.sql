
CREATE OR REPLACE FUNCTION marie_scheduler.fetch_next_job(
    job_name   text,
    batch_size integer DEFAULT 1
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
      WHERE name=job_name AND state IN ('created','retry')
      ORDER BY job_level ASC, priority DESC
      LIMIT 12000
    ),
    active_jobs AS (
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
      GROUP BY dfa.job_id   -- HashAggregate, no Sort+Unique
    ),
    candidate AS MATERIALIZED (
      SELECT aj.id, aj.job_level, aj.priority
      FROM active_jobs aj
      LEFT JOIN open_deps_subset od ON od.job_id = aj.id
      WHERE od.job_id IS NULL
      LIMIT 5000
    )
    SELECT j.*
    FROM candidate c
    JOIN marie_scheduler.job j ON j.id = c.id
$$;

