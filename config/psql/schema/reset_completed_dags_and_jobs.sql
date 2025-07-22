CREATE OR REPLACE FUNCTION marie_scheduler.reset_completed_dags_and_jobs()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_count  INTEGER;
    job_count  INTEGER;
BEGIN
    -- Reset completed DAGs
    UPDATE marie_scheduler.dag
    SET
        state = 'created',
        started_on = NULL,
        completed_on = NULL
    WHERE state = 'completed';
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    -- Reset jobs that are completed and linked to the reset DAGs
    UPDATE marie_scheduler.job
    SET
        state = 'created',
        started_on = NULL,
        completed_on = NULL
    WHERE state = 'completed'
      AND dag_id IN (
          SELECT id FROM marie_scheduler.dag WHERE state = 'created'
      );
    GET DIAGNOSTICS job_count = ROW_COUNT;

    RAISE NOTICE 'Reset % completed DAG(s) and % job(s) to created state.', dag_count, job_count;
END;
$$;
