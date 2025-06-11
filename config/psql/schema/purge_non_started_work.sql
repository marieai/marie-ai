CREATE OR REPLACE FUNCTION marie_scheduler.purge_non_started_work()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_job_count INTEGER;
    deleted_dag_count INTEGER;
BEGIN
    -- Delete jobs whose parent DAG is in state 'created', never started,
    -- and has no job that has ever started.
    DELETE FROM marie_scheduler.job j
    USING marie_scheduler.dag d
    WHERE j.dag_id = d.id
      AND d.state = 'created'
      AND d.started_on IS NULL
      AND NOT EXISTS (
          SELECT 1
            FROM marie_scheduler.job j2
           WHERE j2.dag_id = d.id
             AND (j2.state <> 'created' OR j2.started_on IS NOT NULL)
      );
    GET DIAGNOSTICS deleted_job_count = ROW_COUNT;

    -- Now delete those same DAGs
    DELETE FROM marie_scheduler.dag d
    WHERE d.state = 'created'
      AND d.started_on IS NULL
      AND NOT EXISTS (
          SELECT 1
            FROM marie_scheduler.job j2
           WHERE j2.dag_id = d.id
             AND (j2.state <> 'created' OR j2.started_on IS NOT NULL)
      );
    GET DIAGNOSTICS deleted_dag_count = ROW_COUNT;

    RAISE NOTICE 'Purged % non-started job(s) and % non-started DAG(s).',
                 deleted_job_count,
                 deleted_dag_count;
END;
$$;