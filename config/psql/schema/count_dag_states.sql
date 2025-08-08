CREATE OR REPLACE FUNCTION marie_scheduler.count_dag_states()
RETURNS TABLE(name text, state text, size bigint)
STABLE PARALLEL SAFE LANGUAGE sql AS $$
    WITH j_uniq AS (
      SELECT DISTINCT ON (dag_id, name) dag_id, name
      FROM marie_scheduler.job
      ORDER BY dag_id, name
    )
    SELECT j.name, d.state, COUNT(*)::bigint
    FROM j_uniq j
    JOIN LATERAL (
      SELECT state FROM marie_scheduler.dag WHERE id = j.dag_id
    ) d ON true
    GROUP BY j.name, d.state;
$$;

alter function marie_scheduler.count_dag_states() owner to postgres;

-- NEED TO ADD THIS TO OUR CONFIG
-- SET enable_partitionwise_aggregate = on;
-- SET enable_partitionwise_join = on;
