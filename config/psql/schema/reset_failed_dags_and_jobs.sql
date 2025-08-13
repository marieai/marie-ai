CREATE OR REPLACE FUNCTION marie_scheduler.reset_failed_dags_and_jobs()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_count  INTEGER;
    job_count  INTEGER;
BEGIN
    -- Reset failed DAGs
    UPDATE marie_scheduler.dag
    SET
        state        = 'created',
        started_on   = NULL,
        completed_on = NULL
    WHERE state  IN ('failed', 'retry');
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    -- Reset failed jobs
    UPDATE marie_scheduler.job
    SET
        state        = 'created',
        started_on   = NULL,
        completed_on = NULL
    WHERE state  IN ('failed', 'retry');
    GET DIAGNOSTICS job_count = ROW_COUNT;

    RAISE NOTICE 'Reset % failed DAG(s) and % failed job(s) to created.', dag_count, job_count;
END;
$$