CREATE OR REPLACE FUNCTION marie_scheduler.delete_failed_dags_and_jobs()
RETURNS void
LANGUAGE plpgsql

AS $$

DECLARE
    deleted_job_count  INTEGER;
    deleted_dag_count  INTEGER;
BEGIN
    -- Delete all failed jobs first (to avoid FK violations)
    DELETE FROM marie_scheduler.job
    WHERE state = 'failed';
    GET DIAGNOSTICS deleted_job_count = ROW_COUNT;

    -- Now delete all failed DAGs
    DELETE FROM marie_scheduler.dag
    WHERE state = 'failed';
    GET DIAGNOSTICS deleted_dag_count = ROW_COUNT;

    RAISE NOTICE 'Deleted % failed job(s) and % failed DAG(s).',
                 deleted_job_count,
                 deleted_dag_count;
END;
$$;
