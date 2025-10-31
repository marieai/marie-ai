CREATE OR REPLACE FUNCTION marie_scheduler.count_dag_states()
RETURNS TABLE(name text, state text, size bigint)
STABLE
PARALLEL SAFE
LANGUAGE sql
-- Give this statement enough memory to avoid hashagg spill:
SET work_mem = '256MB'
AS $$
  WITH j_uniq AS (
    SELECT dag_id, name
    FROM marie_scheduler.job
    GROUP BY dag_id, name
  )
  SELECT j.name, d.state, COUNT(*)::bigint
  FROM j_uniq j
  JOIN marie_scheduler.dag d
    ON d.id = j.dag_id
  GROUP BY j.name, d.state
$$;

-- NEED TO ADD THIS TO OUR CONFIG
-- SET enable_partitionwise_aggregate = on;
-- SET enable_partitionwise_join = on;

