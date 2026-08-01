CREATE OR REPLACE FUNCTION {schema}.hydrate_frontier_jobs(dag_ids uuid[])
RETURNS TABLE (
  dag_id uuid,
  job json
)
LANGUAGE plpgsql
STABLE
SET jit = off
SET work_mem = '16MB'
SET plan_cache_mode = force_custom_plan
AS $$
BEGIN
  RETURN QUERY
  SELECT
    j.dag_id,
    json_build_object(
      'id',                j.id,
      'name',              j.name,
      'priority',          j.priority,
      'state',             j.state,
      'retry_limit',       j.retry_limit,
      'start_after',       j.start_after,
      'expire_in_seconds', EXTRACT(EPOCH FROM j.expire_in)::integer,
      'data',              j.data,
      'retry_delay',       j.retry_delay,
      'retry_backoff',     j.retry_backoff,
      'keep_until',        j.keep_until,
      'job_level',         j.job_level,
      'soft_sla',          j.soft_sla,
      'hard_sla',          j.hard_sla,
      'dependencies',      COALESCE(dep.deps, '[]'::json)
    ) AS job
  FROM (
    SELECT DISTINCT requested.dag_id
    FROM unnest(COALESCE(dag_ids, ARRAY[]::uuid[])) AS requested(dag_id)
  ) requested
  JOIN {schema}.job j
    ON j.dag_id = requested.dag_id
  LEFT JOIN LATERAL (
    SELECT json_agg(jd.depends_on_id) FILTER (
             WHERE p.id IS NOT NULL
               AND p.state NOT IN ('completed','failed','cancelled','skipped')
           ) AS deps
    FROM {schema}.job_dependencies jd
    LEFT JOIN {schema}.job p
           ON p.name = jd.depends_on_name
          AND p.id = jd.depends_on_id
    WHERE jd.job_name = j.name
      AND jd.job_id = j.id
  ) dep ON TRUE
  WHERE j.state IN ('created','retry')
  ORDER BY j.dag_id, j.job_level, j.created_on;
END;
$$;
