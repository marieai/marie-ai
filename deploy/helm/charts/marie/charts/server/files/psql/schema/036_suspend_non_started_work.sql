CREATE OR REPLACE FUNCTION {schema}.suspend_non_started_work()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_count  INTEGER;
    job_count  INTEGER;
BEGIN
    -- Suspend non-started DAGs
    UPDATE {schema}.dag
    SET
        state = 'suspended'
    WHERE state IN ('created', 'pending');
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    -- Suspend non-started jobs
    UPDATE {schema}.job
    SET
        state = 'suspended'
    WHERE state IN ('created', 'pending');
    GET DIAGNOSTICS job_count = ROW_COUNT;

    RAISE NOTICE 'Suspended % DAG(s) and % job(s).', dag_count, job_count;
END;
$$