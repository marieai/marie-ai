CREATE OR REPLACE FUNCTION {schema}.delete_orphaned_jobs()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_job_count INTEGER;
BEGIN
    -- Delete any job that has no valid dag reference
    DELETE FROM {schema}.job j
     WHERE j.dag_id IS NULL
        OR NOT EXISTS (
            SELECT 1
              FROM {schema}.dag d
             WHERE d.id = j.dag_id
        );
    GET DIAGNOSTICS deleted_job_count = ROW_COUNT;

    RAISE NOTICE 'Deleted % orphaned job(s).', deleted_job_count;
END;
$$