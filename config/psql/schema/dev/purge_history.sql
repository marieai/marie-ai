-- =========================================================
-- Purge history by age (PG 17-compatible; uses ctid batching)
-- =========================================================
CREATE OR REPLACE FUNCTION marie_scheduler.purge_history(
  p_job_history_keep interval DEFAULT interval '30 days',
  p_dag_history_keep interval DEFAULT interval '30 days',
  p_limit integer DEFAULT 20000
) RETURNS TABLE(table_name text, purged bigint)
LANGUAGE plpgsql AS $$
DECLARE
  v_cnt bigint;
BEGIN
  -- JOB HISTORY
  LOOP
    WITH dels AS (
      SELECT ctid
      FROM marie_scheduler.job_history
      WHERE history_created_on < now() - p_job_history_keep
      ORDER BY history_created_on ASC
      LIMIT p_limit
    )
    DELETE FROM marie_scheduler.job_history j
    USING dels d
    WHERE j.ctid = d.ctid;

    GET DIAGNOSTICS v_cnt = ROW_COUNT;
    table_name := 'job_history'; purged := v_cnt; RETURN NEXT;
    EXIT WHEN v_cnt = 0;
  END LOOP;

  -- DAG HISTORY
  LOOP
    WITH dels AS (
      SELECT ctid
      FROM marie_scheduler.dag_history
      WHERE history_created_on < now() - p_dag_history_keep
      ORDER BY history_created_on ASC
      LIMIT p_limit
    )
    DELETE FROM marie_scheduler.dag_history dgh
    USING dels d
    WHERE dgh.ctid = d.ctid;

    GET DIAGNOSTICS v_cnt = ROW_COUNT;
    table_name := 'dag_history'; purged := v_cnt; RETURN NEXT;
    EXIT WHEN v_cnt = 0;
  END LOOP;
END$$;
