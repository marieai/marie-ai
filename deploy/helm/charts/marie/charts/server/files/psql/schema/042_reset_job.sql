-- sql
-- File: `config/psql/reset_job.sql`
-- Purpose:
--   Reset a single job and its parent DAG to the 'created' state so the job
--   can be reprocessed. This operation clears execution residue for the
--   targeted job and its containing DAG.
--
-- Behavior:
--   - If the supplied job id does not exist, the function returns early and emits a NOTICE.
--   - The function updates only the single job (by id) and the DAG row referenced by that job.
--   - The job row is returned to a fresh schedulable state:
--       state -> 'created'
--       started_on/completed_on -> NULL
--       created_on/start_after -> now()
--       retry_count -> 0
--       output/duration/branch_metadata -> NULL
--       sla_miss_logged -> FALSE
--       lease_owner/lease_expires_at/run_owner/run_lease_expires_at -> NULL
--       lease_epoch -> 0
--   - The parent DAG row is returned to a fresh workflow state:
--       state -> 'created'
--       started_on/completed_on -> NULL
--       created_on/updated_on -> now()
--       duration -> NULL
--       sla_miss_logged -> FALSE
--   - The function emits a NOTICE summarizing affected row counts.
--
-- Usage:
--   SELECT {schema}.reset_job('06904972-5932-7dca-8000-36cda241d09e'::uuid);
--
-- Notes / Considerations:
--   - This does not modify dependent job rows, task history, or external
--     artifacts such as worker KV state. Use with care in production.

CREATE OR REPLACE FUNCTION {schema}.reset_job(p_job_id uuid)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    v_dag_id  uuid;
    dag_count INTEGER := 0;
    job_count INTEGER := 0;
BEGIN
    -- Find the parent dag for the provided job id.
    SELECT dag_id INTO v_dag_id FROM {schema}.job WHERE id = p_job_id;
    IF NOT FOUND THEN
        -- Job not present: nothing to reset.
        RAISE NOTICE 'Job % not found; nothing to reset.', p_job_id;
        RETURN;
    END IF;

    -- Reset the specified job row to a fresh 'created' state so it can be requeued.
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
        run_attempt_id = NULL,
        run_lease_expires_at = NULL
    WHERE id = p_job_id;
    GET DIAGNOSTICS job_count = ROW_COUNT;

    -- Reset the parent DAG row to 'created' as well (single-row update).
    UPDATE {schema}.dag
    SET
        state = 'created',
        started_on = NULL,
        created_on = now(),
        completed_on = NULL,
        updated_on = now(),
        duration = NULL,
        sla_miss_logged = FALSE
    WHERE id = v_dag_id;
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    -- Summary notice for operators
    RAISE NOTICE 'Reset % job(s) and % dag(s) for job_id %.', job_count, dag_count, p_job_id;
END;
$$;

COMMENT ON FUNCTION {schema}.reset_job(uuid) IS
$$
Reset a single job and its parent DAG to a fresh rerunnable state.

Parameters:
  p_job_id uuid - target job id to reset.

Effect:
  - Clears scheduler execution residue for the target job:
      state => 'created'
      started_on/completed_on => NULL
      created_on/start_after => now()
      retry_count => 0
      output/duration/branch_metadata => NULL
      sla_miss_logged => FALSE
      lease_owner/lease_expires_at/run_owner/run_lease_expires_at => NULL
      lease_epoch => 0
  - Resets the parent DAG row:
      state => 'created'
      started_on/completed_on => NULL
      created_on/updated_on => now()
      duration => NULL
      sla_miss_logged => FALSE

Usage example:
  SELECT {schema}.reset_job('06904972-5932-7dca-8000-36cda241d09e'::uuid);

Caution:
  This function does not touch other jobs, dependency graphs, or external
  executor state. Worker KV cleanup or runtime cancellation still needs to
  be handled by the caller when those integrations are active.
$$;
