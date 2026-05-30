CREATE OR REPLACE FUNCTION {schema}.hydrate_frontier_dags()
    RETURNS TABLE
            (
                dag_id         uuid,
                serialized_dag jsonb
            )
    LANGUAGE sql
    STABLE
AS
$$
SELECT d.id AS dag_id,
       d.serialized_dag
FROM {schema}.dag d
WHERE d.id IN (SELECT DISTINCT j.dag_id
               FROM {schema}.job j
               WHERE j.state IN ('created', 'retry'))
ORDER BY d.id;
$$;
