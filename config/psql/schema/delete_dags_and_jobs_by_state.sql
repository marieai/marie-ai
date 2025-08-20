CREATE OR REPLACE FUNCTION marie_scheduler.delete_dags_and_jobs_by_state(
  p_state       text,
  p_job_name    text,
  p_batch_size  integer DEFAULT 50000
)
RETURNS TABLE (jobs_deleted bigint, dags_deleted bigint)
LANGUAGE plpgsql
STRICT
AS $$
DECLARE
  v_batch_count bigint := 0;
  v_total_jobs  bigint := 0;
  v_total_dags  bigint := 0;
BEGIN
  IF p_batch_size <= 0 THEN
    RAISE EXCEPTION 'p_batch_size must be > 0';
  END IF;

  -- Delete matching jobs in batches via (ctid, tableoid)
  LOOP
    WITH dag_ids AS MATERIALIZED (
      SELECT id
      FROM marie_scheduler.dag
      WHERE state::text = p_state
    ),
    job_targets AS MATERIALIZED (
      SELECT j.ctid, j.tableoid
      FROM marie_scheduler.job j
      JOIN dag_ids d ON d.id = j.dag_id
      WHERE j.name = p_job_name
      LIMIT p_batch_size
    )
    DELETE FROM marie_scheduler.job j
    USING job_targets t
    WHERE j.ctid = t.ctid
      AND j.tableoid = t.tableoid;

    GET DIAGNOSTICS v_batch_count = ROW_COUNT;
    v_total_jobs := v_total_jobs + v_batch_count;

    EXIT WHEN v_batch_count = 0;  -- done
  END LOOP;

  -- Always delete DAGs in that state that are now empty
  DELETE FROM marie_scheduler.dag d
  WHERE d.state::text = p_state
    AND NOT EXISTS (
      SELECT 1 FROM marie_scheduler.job j
      WHERE j.dag_id = d.id
    );
  GET DIAGNOSTICS v_total_dags = ROW_COUNT;

  jobs_deleted := v_total_jobs;
  dags_deleted := v_total_dags;
  RETURN NEXT;
END;
$$;
