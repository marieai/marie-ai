-- =========================================================
-- LIVE TABLE CLEANUP (no history writes)
--  - removes finished/expired JOB rows from partitions
--  - removes completed/failed/cancelled DAG rows (when empty or old)
--  - batched via ctid to avoid long locks; time-bounded
-- =========================================================

--     How to use
--
--     Dry run (peek):
--     SELECT count(*) FROM marie_scheduler.jobs_cleanup_candidates;
--     SELECT count(*) FROM marie_scheduler.dags_cleanup_candidates;
--
--     Execute once:
--     SELECT * FROM marie_scheduler.cleanup_jobs_and_dags('14 days','7 days', 5000, 2000, 60000);
--
--     Nightly (via pg_cron or OS cron):
--     SELECT marie_scheduler.cleanup_jobs_and_dags();

SET search_path = public, marie_scheduler, pg_catalog;

-- Delete helper via ctid (still the best way in PG17 for batch deletes)
CREATE OR REPLACE FUNCTION marie_scheduler._delete_ctid_batch(
  p_table regclass,
  p_ctid_sql text
) RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE
  v_sql text;
  v_rows bigint;
BEGIN
  v_sql := format(
    'WITH dels AS (%s)
     DELETE FROM %s t
     USING dels d
     WHERE t.ctid = d.ctid',
    p_ctid_sql, p_table::text
  );
  EXECUTE v_sql;
  GET DIAGNOSTICS v_rows = ROW_COUNT;
  RETURN v_rows;
END$$;

-- Jobs: optionally enforce that a history row exists before delete
CREATE OR REPLACE FUNCTION marie_scheduler.cleanup_jobs(
  p_job_keep interval DEFAULT interval '14 days',
  p_batch_rows integer DEFAULT 20000,        -- PG17 can handle bigger batches
  p_max_ms integer DEFAULT 45000,
  p_require_history boolean DEFAULT false    -- set true if you want a guard
) RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE
  v_start timestamptz := clock_timestamp();
  v_deleted bigint := 0;
  r_part record;
  v_sql text;
  v_batch bigint;
  v_guard text := CASE WHEN p_require_history
                       THEN 'AND EXISTS (SELECT 1 FROM job_history h WHERE h.id = j.id)'
                       ELSE '' END;
BEGIN
  PERFORM set_config('maintenance_work_mem','2GB', false);  -- PG17 can use more

  FOR r_part IN
    SELECT c.oid::regclass AS part
    FROM pg_class c
    JOIN pg_inherits i ON i.inhrelid = c.oid
    JOIN pg_class p ON p.oid = i.inhparent
    WHERE p.relname = 'job' AND c.relkind='r'
  LOOP
    LOOP
      v_sql := format($q$
        SELECT ctid
        FROM %s j
        WHERE (
                j.state IN ('completed','failed')
             OR (j.keep_until IS NOT NULL AND j.keep_until < now())
             OR (j.completed_on IS NOT NULL AND j.completed_on < now() - %L::interval)
              )
              %s
        ORDER BY j.completed_on NULLS LAST, j.keep_until NULLS LAST
        LIMIT %s
      $q$, r_part.part::text, p_job_keep::text, v_guard, p_batch_rows);

      v_batch := marie_scheduler._delete_ctid_batch(r_part.part, v_sql);
      v_deleted := v_deleted + v_batch;

      EXIT WHEN v_batch = 0
         OR (clock_timestamp() - v_start) > make_interval(secs => p_max_ms/1000.0);
    END LOOP;

    EXIT WHEN (clock_timestamp() - v_start) > make_interval(secs => p_max_ms/1000.0);
  END LOOP;

  -- PG17: vacuum can use more memory; do index cleanup & truncate if possible
  EXECUTE 'VACUUM (ANALYZE, INDEX_CLEANUP ON, TRUNCATE ON) job';

  RETURN v_deleted;
END$$;

-- Dags: delete finished dags that are old OR have no non-finished jobs
CREATE OR REPLACE FUNCTION marie_scheduler.cleanup_dags(
  p_dag_keep interval DEFAULT interval '7 days',
  p_batch_rows integer DEFAULT 10000,
  p_max_ms integer DEFAULT 30000,
  p_require_history boolean DEFAULT false
) RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE
  v_start timestamptz := clock_timestamp();
  v_deleted bigint := 0;
  v_sql text;
  v_batch bigint;
  v_guard text := CASE WHEN p_require_history
                       THEN 'AND EXISTS (SELECT 1 FROM dag_history h WHERE h.id = d.id)'
                       ELSE '' END;
BEGIN
  PERFORM set_config('maintenance_work_mem','2GB', false);

  LOOP
    v_sql := format($q$
      SELECT ctid
      FROM dag d
      WHERE d.state IN ('completed','failed','cancelled')
        AND (
             (d.completed_on IS NOT NULL AND d.completed_on < now() - %L::interval)
             OR NOT EXISTS (
                 SELECT 1 FROM job j
                 WHERE j.dag_id = d.id AND j.state NOT IN ('completed','failed')
               )
            )
        %s
      ORDER BY d.completed_on NULLS LAST
      LIMIT %s
    $q$, p_dag_keep::text, v_guard, p_batch_rows);

    v_batch := marie_scheduler._delete_ctid_batch('dag', v_sql);
    v_deleted := v_deleted + v_batch;

    EXIT WHEN v_batch = 0
       OR (clock_timestamp() - v_start) > make_interval(secs => p_max_ms/1000.0);
  END LOOP;

  EXECUTE 'VACUUM (ANALYZE, INDEX_CLEANUP ON, TRUNCATE ON) dag';

  RETURN v_deleted;
END$$;

-- Wrapper
CREATE OR REPLACE FUNCTION marie_scheduler.cleanup_jobs_and_dags(
  p_job_keep interval DEFAULT interval '14 days',
  p_dag_keep interval DEFAULT interval '7 days',
  p_job_batch integer DEFAULT 20000,
  p_dag_batch integer DEFAULT 10000,
  p_max_ms integer DEFAULT 90000,
  p_require_history boolean DEFAULT false
) RETURNS TABLE(step text, deleted bigint)
LANGUAGE plpgsql AS $$
DECLARE
  v_each_ms integer := GREATEST(15000, p_max_ms/2);
  v_jobs bigint;
  v_dags bigint;
BEGIN
  v_jobs := marie_scheduler.cleanup_jobs(p_job_keep, p_job_batch, v_each_ms, p_require_history);
  step := 'jobs'; deleted := v_jobs; RETURN NEXT;

  v_dags := marie_scheduler.cleanup_dags(p_dag_keep, p_dag_batch, v_each_ms, p_require_history);
  step := 'dags'; deleted := v_dags; RETURN NEXT;
END$$;

-- Quick candidate views (unchanged)
CREATE OR REPLACE VIEW marie_scheduler.jobs_cleanup_candidates AS
SELECT j.*
FROM job j
WHERE
  j.state IN ('completed','failed')
  OR (j.keep_until IS NOT NULL AND j.keep_until < now())
  OR (j.completed_on IS NOT NULL AND j.completed_on < now() - interval '14 days');

CREATE OR REPLACE VIEW marie_scheduler.dags_cleanup_candidates AS
SELECT d.*
FROM dag d
WHERE d.state IN ('completed','failed','cancelled')
  AND (
    (d.completed_on IS NOT NULL AND d.completed_on < now() - interval '7 days')
    OR NOT EXISTS (SELECT 1 FROM job j WHERE j.dag_id = d.id AND j.state NOT IN ('completed','failed'))
  );
