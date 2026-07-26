-- Reset every DAG and its jobs to a fresh schedulable state.
CREATE OR REPLACE FUNCTION {schema}.reset_all()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_count INTEGER;
    job_count INTEGER;
BEGIN
    UPDATE {schema}.job
    SET state = 'created',
        started_on = NULL,
        created_on = now(),
        completed_on = NULL,
        start_after = now(),
        retry_count = 0,
        output = NULL,
        duration = NULL,
        sla_miss_logged = FALSE,
        branch_metadata = NULL,
        lease_owner = NULL,
        lease_expires_at = NULL,
        lease_epoch = 0,
        run_owner = NULL,
        run_attempt_id = NULL,
        run_lease_expires_at = NULL
    WHERE dag_id IN (SELECT id FROM {schema}.dag);
    GET DIAGNOSTICS job_count = ROW_COUNT;

    UPDATE {schema}.dag
    SET state = 'created',
        started_on = NULL,
        created_on = now(),
        completed_on = NULL,
        updated_on = now(),
        duration = NULL,
        sla_miss_logged = FALSE;
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    RAISE NOTICE 'Reset % DAG(s) and % job(s) to a fresh schedulable state.',
        dag_count, job_count;
END;
$$;

COMMENT ON FUNCTION {schema}.reset_all() IS
'Reset every DAG and its jobs for rerun while preserving dependency and attempt audit history.';
