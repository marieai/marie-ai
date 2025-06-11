CREATE OR REPLACE FUNCTION marie_scheduler.reset_active_dags_and_jobs()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_count  INTEGER;
    job_count  INTEGER;
BEGIN
    -- Reset active DAGs
    UPDATE marie_scheduler.dag
    SET
        state        = 'created',
        started_on   = NULL,
        completed_on = NULL
    WHERE state = 'active';
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    -- Reset active jobs (for those DAGs)
    UPDATE marie_scheduler.job
    SET
        state        = 'created',
        started_on   = NULL,
        completed_on = NULL
    WHERE state = 'active';
    GET DIAGNOSTICS job_count = ROW_COUNT;

    RAISE NOTICE 'Reset % DAG(s) and % job(s) to created.', dag_count, job_count;
END;
$$