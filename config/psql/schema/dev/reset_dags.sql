-- sql
-- File: `config/psql/schema/reset_dags.sql`
-- Purpose:
--   Reset one or more DAGs and all jobs that belong to them to the 'created'
--   state so the DAG(s) can be reprocessed with fresh scheduler state.
--
-- Behavior:
--   - If `p_dag_ids` is NULL or empty the function returns immediately with a NOTICE.
--   - Job rows are returned to a fresh schedulable state:
--       * `state`       => 'created'
--       * `started_on`  => NULL
--       * `created_on`  => now()
--       * `completed_on`=> NULL
--       * `start_after` => now()
--       * `retry_count` => 0
--       * `output`/`duration`/`branch_metadata` => NULL
--       * `sla_miss_logged` => FALSE
--       * `lease_owner`/`lease_expires_at`/`run_owner`/`run_lease_expires_at` => NULL
--       * `lease_epoch` => 0
--   - DAG rows are returned to a fresh workflow state:
--       * `state`       => 'created'
--       * `started_on`  => NULL
--       * `created_on`  => now()
--       * `completed_on`=> NULL
--       * `updated_on`  => now()
--       * `duration`    => NULL
--       * `sla_miss_logged` => FALSE
--   - Emits a NOTICE summarizing the number of job and DAG rows affected.
--
-- Safety / Considerations:
--   - This is an in-database scheduler reset. It does not:
--       * modify dependency graph tables,
--       * clear worker KV or external runtime resources,
--       * update job history/audit tables, or
--       * notify external schedulers/executors.
--   - Use with care in production. If additional cleanup is required (leases,
--     caches, frontiers, notifications), perform those steps in your operational
--     workflow after calling this function.
--
-- Usage:
--   SELECT marie_scheduler.reset_dags(ARRAY[
--     '06904972-5932-7dca-8000-36cda241d087'::uuid,
--     '01234567-89ab-cdef-0123-456789abcdef'::uuid
--   ]);
--
-- Example (no-op):
--   SELECT marie_scheduler.reset_dags(NULL);
--   -- Emits: NOTICE 'No DAG ids provided; nothing to reset.'
--
-- Implementation:
CREATE OR REPLACE FUNCTION marie_scheduler.reset_dags(p_dag_ids uuid[])
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dag_count   INTEGER := 0;
    job_count   INTEGER := 0;
    input_count INTEGER := COALESCE(array_length(p_dag_ids, 1), 0);
BEGIN
    -- No-op if nothing supplied
    IF input_count = 0 THEN
        RAISE NOTICE 'No DAG ids provided; nothing to reset.';
        RETURN;
    END IF;

    -- Reset all jobs belonging to the supplied DAG ids
    UPDATE marie_scheduler.job
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
    WHERE dag_id = ANY(p_dag_ids);
    GET DIAGNOSTICS job_count = ROW_COUNT;

    -- Reset the DAG rows
    UPDATE marie_scheduler.dag
    SET
        state = 'created',
        started_on = NULL,
        created_on = now(),
        completed_on = NULL,
        updated_on = now(),
        duration = NULL,
        sla_miss_logged = FALSE
    WHERE id = ANY(p_dag_ids);
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    RAISE NOTICE 'Reset % job(s) and % dag(s) for % provided dag_id(s).', job_count, dag_count, input_count;
END;
$$;

COMMENT ON FUNCTION marie_scheduler.reset_dags(uuid[]) IS
$$
Reset multiple DAGs and all their jobs to a fresh rerunnable state.

Parameters:
  p_dag_ids uuid[] - array of DAG ids to reset.

Effect:
  - Clears scheduler execution residue for all jobs in the supplied DAGs:
      state => 'created'
      started_on/completed_on => NULL
      created_on/start_after => now()
      retry_count => 0
      output/duration/branch_metadata => NULL
      sla_miss_logged => FALSE
      lease_owner/lease_expires_at/run_owner/run_lease_expires_at => NULL
      lease_epoch => 0
  - Resets the DAG rows:
      state => 'created'
      started_on/completed_on => NULL
      created_on/updated_on => now()
      duration => NULL
      sla_miss_logged => FALSE

Behavior:
  - If `p_dag_ids` is NULL or empty the function returns early with a NOTICE.
  - Emits a NOTICE summarizing affected row counts.

Usage example:
  SELECT marie_scheduler.reset_dags(ARRAY[
    '06904972-5932-7dca-8000-36cda241d087'::uuid,
    '...another-dag-id...'::uuid
  ]);

Caution:
  - This does not clear dependency graphs, job history tables, worker KV state,
    or external executor artifacts. Additional cleanup may still be required.
$$;
