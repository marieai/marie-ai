-- This function resets the state of active DAGs and their associated jobs to 'created'.
-- Usage:
-- Single job name
-- SELECT reset_active_dags_and_jobs(ARRAY['gen5_extract']);
-- Multiple job names
-- SELECT reset_active_dags_and_jobs(ARRAY['gen5_extract', 'gen4_validate']);

CREATE OR REPLACE FUNCTION {schema}.reset_active_dags_and_jobs(p_job_names TEXT[])
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_count  INTEGER;
    job_count  INTEGER;
BEGIN
    -- Reset active jobs with name in list
    UPDATE {schema}.job
    SET
        state        = 'created',
        started_on   = NULL,
        completed_on = NULL
    WHERE state = 'active'
      AND name = ANY(p_job_names);
    GET DIAGNOSTICS job_count = ROW_COUNT;

    -- Reset DAGs associated with those jobs
    UPDATE {schema}.dag
    SET
        state        = 'created',
        started_on   = NULL,
        completed_on = NULL
    WHERE state = 'active'
      AND id IN (
          SELECT DISTINCT dag_id
          FROM {schema}.job
          WHERE name = ANY(p_job_names)
      );
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    RAISE NOTICE 'Reset % job(s) and % DAG(s) to created for job name(s): %.', job_count, dag_count, p_job_names;
END;
$$