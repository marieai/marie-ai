CREATE OR REPLACE FUNCTION {schema}.count_job_states()
RETURNS TABLE(name TEXT, state TEXT, size BIGINT)
LANGUAGE sql
STABLE
AS $$
    SELECT name, state, count(*) as size
    FROM {schema}.job
    GROUP BY name, state
    ORDER BY name, state;
$$;


