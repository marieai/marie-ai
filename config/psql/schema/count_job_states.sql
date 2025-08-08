CREATE OR REPLACE FUNCTION marie_scheduler.count_job_states()
RETURNS TABLE(name TEXT, state TEXT, size BIGINT)
LANGUAGE sql
STABLE
AS $$
    SELECT name, state, count(*) as size
    FROM marie_scheduler.job
    GROUP BY name, state
    ORDER BY name, state;
$$;


