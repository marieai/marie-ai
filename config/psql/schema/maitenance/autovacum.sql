-- Helper to apply reloptions to leaf partitions (or the table itself if not partitioned)
DO $$
DECLARE
  tgt       regclass;
  relk      "char";
  part      regclass;
  opts_job  text := 'autovacuum_vacuum_scale_factor = 0.02,
                     autovacuum_vacuum_threshold    = 500,
                     autovacuum_analyze_scale_factor= 0.05,
                     autovacuum_analyze_threshold   = 500';
  opts_dag  text := 'autovacuum_vacuum_scale_factor = 0.1,
                     autovacuum_analyze_scale_factor= 0.1';
BEGIN
  -- ========== JOB ==========
  tgt := 'marie_scheduler.job'::regclass;
  SELECT relkind INTO relk FROM pg_class WHERE oid = tgt;

  IF relk = 'p' THEN
    -- Apply to all leaf partitions of job
    FOR part IN
      SELECT c.oid::regclass
      FROM pg_class c
      JOIN pg_inherits i ON i.inhrelid = c.oid
      WHERE i.inhparent = tgt
        AND c.relispartition
        AND NOT EXISTS (SELECT 1 FROM pg_inherits ch WHERE ch.inhparent = c.oid)  -- leaf only
    LOOP
      EXECUTE format('ALTER TABLE %s SET (%s)', part, opts_job);
    END LOOP;
  ELSE
    EXECUTE format('ALTER TABLE %s SET (%s)', tgt, opts_job);
  END IF;

  -- ========== DAG ==========
  tgt := 'marie_scheduler.dag'::regclass;
  SELECT relkind INTO relk FROM pg_class WHERE oid = tgt;

  IF relk = 'p' THEN
    FOR part IN
      SELECT c.oid::regclass
      FROM pg_class c
      JOIN pg_inherits i ON i.inhrelid = c.oid
      WHERE i.inhparent = tgt
        AND c.relispartition
        AND NOT EXISTS (SELECT 1 FROM pg_inherits ch WHERE ch.inhparent = c.oid)
    LOOP
      EXECUTE format('ALTER TABLE %s SET (%s)', part, opts_dag);
    END LOOP;
  ELSE
    EXECUTE format('ALTER TABLE %s SET (%s)', tgt, opts_dag);
  END IF;
END $$;

-- History tables (likely not partitioned) can be altered directly:
ALTER TABLE marie_scheduler.job_history SET (autovacuum_vacuum_scale_factor = 0.2);
ALTER TABLE marie_scheduler.dag_history SET (autovacuum_vacuum_scale_factor = 0.2);




-- Apply aggressive settings to each job partition
DO $$
DECLARE
  part regclass;
BEGIN
  FOR part IN
    SELECT c.oid::regclass
    FROM pg_class c
    JOIN pg_inherits i ON i.inhrelid = c.oid
    JOIN pg_class p ON p.oid = i.inhparent
    WHERE p.relname = 'job'
      AND c.relkind = 'r'
  LOOP
    EXECUTE format($sql$
      ALTER TABLE %s SET (
        autovacuum_vacuum_scale_factor = 0.01,
        autovacuum_vacuum_threshold    = 200,
        autovacuum_analyze_scale_factor= 0.02,
        autovacuum_analyze_threshold   = 200
      )
    $sql$, part::text);
  END LOOP;
END$$;