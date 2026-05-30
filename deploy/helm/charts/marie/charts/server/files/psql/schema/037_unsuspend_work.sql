CREATE OR REPLACE FUNCTION {schema}.unsuspend_work()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_count  INTEGER;
    job_count  INTEGER;
BEGIN
    -- Unsuspend DAGs
    UPDATE {schema}.dag
    SET
        state = 'created',
        started_on = NULL,
        completed_on = NULL
    WHERE state = 'suspended';
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    -- Unsuspend jobs
    UPDATE {schema}.job
    SET
        state = 'created',
        started_on = NULL,
        completed_on = NULL
    WHERE state = 'suspended';
    GET DIAGNOSTICS job_count = ROW_COUNT;

    RAISE NOTICE 'Unsuspended % DAG(s) and % job(s) and reset them to created state.', dag_count, job_count;
END;
$$