-- sql
-- File: `config/psql/reset_dag.sql`
-- Purpose:
--   Reset a DAG and all its jobs to the 'created' state so the DAG can be
--   reprocessed. This updates timestamps and clears started/completed markers
--   for the DAG and every job that references it.
--
-- Behavior:
--   - If the supplied dag id does not exist, the function returns early and emits a NOTICE.
--   - Updates all jobs with `dag_id = p_dag_id` and the single DAG row with `id = p_dag_id`.
--   - For each updated row the following columns are set:
--       state -> 'created'
--       started_on -> NULL
--       created_on -> now()
--       completed_on -> NULL
--   - Emits a NOTICE summarizing affected row counts.
--
-- Usage:
--   SELECT marie_scheduler.reset_dag('06904972-5932-7dca-8000-36cda241d087'::uuid);
--
-- Caution:
--   This is a minimal reset. It does not modify dependency rows, job history,
--   leases, or external executor state. Additional cleanup may be required
--   depending on integration.

CREATE OR REPLACE FUNCTION marie_scheduler.reset_dag(p_dag_id uuid)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_exists BOOLEAN;
    dag_count INTEGER := 0;
    job_count INTEGER := 0;
BEGIN
    -- Verify DAG exists
    SELECT EXISTS(SELECT 1 FROM marie_scheduler.dag WHERE id = p_dag_id) INTO dag_exists;
    IF NOT dag_exists THEN
        RAISE NOTICE 'DAG % not found; nothing to reset.', p_dag_id;
        RETURN;
    END IF;

    -- Reset all jobs belonging to the DAG
    UPDATE marie_scheduler.job
    SET
        state = 'created',
        started_on = NULL,
        created_on = now(),
        completed_on = NULL
    WHERE dag_id = p_dag_id;
    GET DIAGNOSTICS job_count = ROW_COUNT;

    -- Reset the DAG row
    UPDATE marie_scheduler.dag
    SET
        state = 'created',
        started_on = NULL,
        created_on = now(),
        completed_on = NULL
    WHERE id = p_dag_id;
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    RAISE NOTICE 'Reset % job(s) and % dag(s) for dag_id %.', job_count, dag_count, p_dag_id;
END;
$$;

COMMENT ON FUNCTION marie_scheduler.reset_dag(uuid) IS
$$
Reset a DAG and all its jobs to the 'created' state.

Parameters:
  p_dag_id uuid - target DAG id to reset.

Effect:
  - Updates all job rows where dag_id = p_dag_id and the corresponding dag row:
      state => 'created'
      started_on => NULL
      created_on => now()
      completed_on => NULL

Usage example:
  SELECT marie_scheduler.reset_dag('06904972-5932-7dca-8000-36cda241d087'::uuid);

Caution:
  This function performs a minimal reset. It does not touch dependency graphs,
  job history tables, lease state, or external executor artifacts. Consider
  additional cleanup (leases, frontier hydration, cache invalidation) if needed.
$$;
