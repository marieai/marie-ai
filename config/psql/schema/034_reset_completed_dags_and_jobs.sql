CREATE OR REPLACE FUNCTION {schema}.reset_completed_dags_and_jobs()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    reset_at TIMESTAMPTZ := statement_timestamp();
    completed_dag_ids UUID[];
    dag_count INTEGER := 0;
    job_count INTEGER := 0;
BEGIN
    SELECT array_agg(completed.id)
    INTO completed_dag_ids
    FROM (
        SELECT id
        FROM {schema}.dag
        WHERE state = 'completed'
        FOR UPDATE
    ) AS completed;

    IF COALESCE(array_length(completed_dag_ids, 1), 0) = 0 THEN
        RAISE NOTICE 'No completed DAGs found; nothing to reset.';
        RETURN;
    END IF;

    UPDATE {schema}.job
    SET
        state = 'created',
        started_on = NULL,
        created_on = reset_at,
        completed_on = NULL,
        start_after = reset_at,
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
    WHERE dag_id = ANY(completed_dag_ids);
    GET DIAGNOSTICS job_count = ROW_COUNT;

    UPDATE {schema}.dag
    SET
        state = 'created',
        started_on = NULL,
        created_on = reset_at,
        completed_on = NULL,
        updated_on = reset_at,
        duration = NULL,
        sla_miss_logged = FALSE
    WHERE id = ANY(completed_dag_ids);
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    RAISE NOTICE 'Reset % completed DAG(s) and % job(s) to a fresh schedulable state.',
        dag_count, job_count;
END;
$$;

COMMENT ON FUNCTION {schema}.reset_completed_dags_and_jobs() IS
'Reset exactly the DAGs that were completed when the function began and clear their scheduler execution residue. Existing history and attempt audit rows are preserved.';
