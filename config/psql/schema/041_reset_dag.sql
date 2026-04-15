-- sql
-- File: `config/psql/reset_dag.sql`
-- Purpose:
--   Reset a DAG and all its jobs to the 'created' state so the DAG can be
--   reprocessed. This clears execution residue on both the DAG row and every
--   job that references it.
--
-- Behavior:
--   - If the supplied dag id does not exist, the function returns early and emits a NOTICE.
--   - Updates all jobs with `dag_id = p_dag_id` and the single DAG row with `id = p_dag_id`.
--   - Job rows are returned to a fresh schedulable state:
--       state -> 'created'
--       started_on/completed_on -> NULL
--       created_on/start_after -> now()
--       retry_count -> 0
--       output/duration/branch_metadata -> NULL
--       sla_miss_logged -> FALSE
--       lease_owner/lease_expires_at/run_owner/run_lease_expires_at -> NULL
--       lease_epoch -> 0
--   - DAG rows are returned to a fresh workflow state:
--       state -> 'created'
--       started_on/completed_on -> NULL
--       created_on/updated_on -> now()
--       duration -> NULL
--       sla_miss_logged -> FALSE
--   - Emits a NOTICE summarizing affected row counts.
--
-- Usage:
--   SELECT {schema}.reset_dag('06904972-5932-7dca-8000-36cda241d087'::uuid);
--
-- Caution:
--   This does not modify dependency rows, job history, or external executor
--   state outside the scheduler database. Worker KV cleanup and runtime
--   cancellation still need to be handled by the caller.

CREATE OR REPLACE FUNCTION {schema}.reset_dag(p_dag_id uuid)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_exists BOOLEAN;
    dag_count INTEGER := 0;
    job_count INTEGER := 0;
BEGIN
    -- Verify DAG exists
    SELECT EXISTS(SELECT 1 FROM {schema}.dag WHERE id = p_dag_id) INTO dag_exists;
    IF NOT dag_exists THEN
        RAISE NOTICE 'DAG % not found; nothing to reset.', p_dag_id;
        RETURN;
    END IF;

    -- Reset all jobs belonging to the DAG
    UPDATE {schema}.job
    SET
        state = 'created',
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
        run_lease_expires_at = NULL
    WHERE dag_id = p_dag_id;
    GET DIAGNOSTICS job_count = ROW_COUNT;

    -- Reset the DAG row
    UPDATE {schema}.dag
    SET
        state = 'created',
        started_on = NULL,
        created_on = now(),
        completed_on = NULL,
        updated_on = now(),
        duration = NULL,
        sla_miss_logged = FALSE
    WHERE id = p_dag_id;
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    RAISE NOTICE 'Reset % job(s) and % dag(s) for dag_id %.', job_count, dag_count, p_dag_id;
END;
$$;

COMMENT ON FUNCTION {schema}.reset_dag(uuid) IS
$$
Reset a DAG and all its jobs to a fresh rerunnable state.

Parameters:
  p_dag_id uuid - target DAG id to reset.

Effect:
  - Clears scheduler execution residue for every job in the DAG:
      state => 'created'
      started_on/completed_on => NULL
      created_on/start_after => now()
      retry_count => 0
      output/duration/branch_metadata => NULL
      sla_miss_logged => FALSE
      lease_owner/lease_expires_at/run_owner/run_lease_expires_at => NULL
      lease_epoch => 0
  - Resets the DAG row:
      state => 'created'
      started_on/completed_on => NULL
      created_on/updated_on => now()
      duration => NULL
      sla_miss_logged => FALSE

Usage example:
  SELECT {schema}.reset_dag('06904972-5932-7dca-8000-36cda241d087'::uuid);

Caution:
  This function does not touch dependency graphs, job history tables, or
  external executor artifacts outside the scheduler database. Consider
  worker KV cleanup and runtime cancellation if those integrations are active.
$$;
