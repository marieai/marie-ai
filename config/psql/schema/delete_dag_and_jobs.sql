CREATE OR REPLACE FUNCTION marie_scheduler.delete_dag_and_jobs(p_dag_id uuid)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_job_count  INTEGER;
    deleted_dag_count  INTEGER;
BEGIN
    -- 1) Delete all jobs for the specified DAG
    DELETE FROM marie_scheduler.job
     WHERE dag_id = p_dag_id;
    GET DIAGNOSTICS deleted_job_count = ROW_COUNT;

    -- 2) Delete the DAG itself
    DELETE FROM marie_scheduler.dag
     WHERE id = p_dag_id;
    GET DIAGNOSTICS deleted_dag_count = ROW_COUNT;

    -- 3) Report how many rows were purged
    RAISE NOTICE 'Deleted % job(s) and % dag(s) for DAG id %',
                 deleted_job_count,
                 deleted_dag_count,
                 p_dag_id;
END;
$$;