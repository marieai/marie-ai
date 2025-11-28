CREATE OR REPLACE FUNCTION {schema}.hydrate_frontier_jobs(dag_ids uuid[])
RETURNS TABLE (
  dag_id uuid,
  job json
)
LANGUAGE sql
STABLE
AS $$
SELECT
  j.dag_id,
  json_build_object(
    'id',            j.id,
    'name',          j.name,
    'priority',      j.priority,
    'state',         j.state,
    'retry_limit',   j.retry_limit,
    'start_after',   j.start_after,
    'expire_in',     j.expire_in,
    'data',          j.data,
    'retry_delay',   j.retry_delay,
    'retry_backoff', j.retry_backoff,
    'keep_until',    j.keep_until,
    'job_level',     j.job_level,
    'dependencies',  COALESCE(dep.deps, '[]'::json)
  ) AS job
FROM {schema}.job j
LEFT JOIN LATERAL (
  SELECT json_agg(jd.depends_on_id) FILTER (
           WHERE p.id IS NOT NULL
             AND p.state NOT IN ('completed','failed','cancelled')
         ) AS deps
  FROM {schema}.job_dependencies jd
  LEFT JOIN {schema}.job p
         ON p.id = jd.depends_on_id
  WHERE jd.job_id = j.id
) dep ON TRUE
WHERE j.dag_id = ANY(dag_ids)
  AND j.state IN ('created','retry')
ORDER BY j.dag_id, j.job_level, j.created_on;
$$;