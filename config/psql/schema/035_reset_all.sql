CREATE OR REPLACE FUNCTION {schema}.reset_all()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_count  INTEGER;
    job_count  INTEGER;
BEGIN
    -- Reset completed DAGs
    UPDATE {schema}.dag
    SET
        state = 'created',
        started_on = NULL,
        completed_on = NULL;
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    -- Reset jobs that are completed and linked to the reset DAGs
    UPDATE {schema}.job
    SET
        state = 'created',
        started_on = NULL,
        completed_on = NULL
    WHERE
       dag_id IN (
          SELECT id FROM {schema}.dag WHERE state = 'created'
      );
    GET DIAGNOSTICS job_count = ROW_COUNT;

    RAISE NOTICE 'Reset % completed DAG(s) and % job(s) to created state.', dag_count, job_count;
END;
$$;