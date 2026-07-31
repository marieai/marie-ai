-- Reset every DAG and its jobs to a fresh schedulable state.
CREATE OR REPLACE FUNCTION {schema}.reset_all()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    reset_at TIMESTAMPTZ := statement_timestamp();
    dag_count BIGINT;
    job_count BIGINT;
BEGIN
    ALTER TABLE {schema}.job
        DISABLE TRIGGER job_update_state_trigger;
    ALTER TABLE {schema}.dag
        DISABLE TRIGGER dag_update_state_trigger;
    ALTER TABLE {schema}.dag
        DISABLE TRIGGER trg_dag_state_changed;

    UPDATE {schema}.job
    SET state = 'created',
        started_on = NULL,
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
    WHERE state IS DISTINCT FROM 'created'
       OR started_on IS NOT NULL
       OR completed_on IS NOT NULL
       OR start_after > reset_at
       OR retry_count <> 0
       OR output IS NOT NULL
       OR duration IS NOT NULL
       OR sla_miss_logged
       OR branch_metadata IS NOT NULL
       OR lease_owner IS NOT NULL
       OR lease_expires_at IS NOT NULL
       OR lease_epoch IS DISTINCT FROM 0
       OR run_owner IS NOT NULL
       OR run_attempt_id IS NOT NULL
       OR run_lease_expires_at IS NOT NULL;
    GET DIAGNOSTICS job_count = ROW_COUNT;

    UPDATE {schema}.dag
    SET state = 'created',
        started_on = NULL,
        completed_on = NULL,
        updated_on = reset_at,
        duration = NULL,
        sla_miss_logged = FALSE
    WHERE state IS DISTINCT FROM 'created'
       OR started_on IS NOT NULL
       OR completed_on IS NOT NULL
       OR duration IS NOT NULL
       OR sla_miss_logged;
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    ALTER TABLE {schema}.job
        ENABLE TRIGGER job_update_state_trigger;
    ALTER TABLE {schema}.dag
        ENABLE TRIGGER dag_update_state_trigger;
    ALTER TABLE {schema}.dag
        ENABLE TRIGGER trg_dag_state_changed;

    RAISE NOTICE 'Reset % DAG(s) and % job(s) to a fresh schedulable state.',
        dag_count, job_count;
END;
$$;

COMMENT ON FUNCTION {schema}.reset_all() IS
'Reset DAGs and jobs with execution residue while preserving existing dependency and audit rows. Run with scheduler writers stopped; bulk resets do not append per-row history or DAG notifications.';
