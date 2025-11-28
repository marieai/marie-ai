-- sql
-- File: `config/psql/schema/reset_dags.sql`
-- Purpose:
--   Reset one or more DAGs and all jobs that belong to them to the 'created'
--   state so the DAG(s) can be reprocessed.
--
-- Behavior:
--   - If `p_dag_ids` is NULL or empty the function returns immediately with a NOTICE.
--   - Updates all job rows where `dag_id = ANY(p_dag_ids)` and all DAG rows where
--     `id = ANY(p_dag_ids)` to:
--       * `state`       => 'created'
--       * `started_on`  => NULL
--       * `created_on`  => now()
--       * `completed_on`=> NULL
--   - Emits a NOTICE summarizing the number of job and DAG rows affected.
--
-- Safety / Considerations:
--   - This is a minimal, in-database reset. It does not:
--       * modify dependency graph tables,
--       * clear executor leases or external resources,
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
        completed_on = NULL
    WHERE dag_id = ANY(p_dag_ids);
    GET DIAGNOSTICS job_count = ROW_COUNT;

    -- Reset the DAG rows
    UPDATE marie_scheduler.dag
    SET
        state = 'created',
        started_on = NULL,
        created_on = now(),
        completed_on = NULL
    WHERE id = ANY(p_dag_ids);
    GET DIAGNOSTICS dag_count = ROW_COUNT;

    RAISE NOTICE 'Reset % job(s) and % dag(s) for % provided dag_id(s).', job_count, dag_count, input_count;
END;
$$;

COMMENT ON FUNCTION marie_scheduler.reset_dags(uuid[]) IS
$$
Reset multiple DAGs and all their jobs to the 'created' state.

Parameters:
  p_dag_ids uuid[] - array of DAG ids to reset.

Effect:
  - Updates all job rows where dag_id = ANY(p_dag_ids) and all DAG rows where id = ANY(p_dag_ids):
      state => 'created'
      started_on => NULL
      created_on => now()
      completed_on => NULL

Behavior:
  - If `p_dag_ids` is NULL or empty the function returns early with a NOTICE.
  - Emits a NOTICE summarizing affected row counts.

Usage example:
  SELECT marie_scheduler.reset_dags(ARRAY[
    '06904972-5932-7dca-8000-36cda241d087'::uuid,
    '...another-dag-id...'::uuid
  ]);

Caution:
  - This is a minimal reset and does not clear leases, frontier state, dependency graphs,
    job history tables, or external executor artifacts. Additional cleanup may be required.
$$;
