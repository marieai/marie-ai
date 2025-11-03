-- sql
-- File: `config/psql/reset_job.sql`
-- Purpose:
--   Reset a single job and its parent DAG to the 'created' state so the job
--   can be reprocessed. This operation updates timestamps and clears started/completed
--   markers for the targeted job and its containing DAG.
--
-- Behavior:
--   - If the supplied job id does not exist, the function returns early and emits a NOTICE.
--   - The function updates only the single job (by id) and the DAG row referenced by that job.
--   - Both job and DAG rows have: state -> 'created', started_on -> NULL,
--     created_on -> NOW(), completed_on -> NULL.
--   - The function emits a NOTICE summarizing affected row counts.
--
-- Usage:
--   SELECT marie_scheduler.reset_job('06904972-5932-7dca-8000-36cda241d09e'::uuid);
--
-- Notes / Considerations:
--   - This is a light-weight reset: it does not modify dependent job rows, task history,
--     or external artifacts. Use with care in production.
--   - Consider additional consistency steps if DAG-level invariants or caches must be refreshed.

CREATE OR REPLACE FUNCTION marie_scheduler.reset_job(p_job_id uuid)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    v_dag_id  uuid;
    dag_count INTEGER := 0;
    job_count INTEGER := 0;
BEGIN
    -- Find the parent dag for the provided job id.
    SELECT dag_id INTO v_dag_id FROM marie_scheduler.job WHERE id = p_job_id;
    IF NOT FOUND THEN
        -- Job not present: nothing to reset.
        RAISE NOTICE 'Job % not found; nothing to reset.', p_job_id;
        RETURN;
    END IF;

    -- Reset the specified job row to a fresh 'created' state so it can be requeued.
    UPDATE marie_scheduler.job
    SET
        state = 'created',
        started_on = NULL,
        created_on = NOW(),
        completed_on = NULL
    WHERE id = p_job_id;
    GET DIAGNOSTICS job_count = ROW_COUNT;

    -- Reset the parent DAG row to 'created' as well (single-row update).
    UPDATE marie_scheduler.dag
    SET
        state = 'created',
        started_on = NULL,
        created_on = NOW(),
        completed_on = NULL
    WHERE id = v_dag_id;
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    -- Summary notice for operators
    RAISE NOTICE 'Reset % job(s) and % dag(s) for job_id %.', job_count, dag_count, p_job_id;
END;
$$;

COMMENT ON FUNCTION marie_scheduler.reset_job(uuid) IS
$$
Reset a single job and its parent DAG to the 'created' state.

Parameters:
  p_job_id uuid - target job id to reset.

Effect:
  - Updates the target job row and its parent dag row:
      state => 'created'
      started_on => NULL
      created_on => NOW()
      completed_on => NULL

Usage example:
  SELECT marie_scheduler.reset_job('06904972-5932-7dca-8000-36cda241d09e'::uuid);

Caution:
  This function performs a minimal reset. It does not touch other jobs,
  dependency graphs, or external executor state. Additional cleanup or
  cache invalidation may be required depending on system integration.
$$;
