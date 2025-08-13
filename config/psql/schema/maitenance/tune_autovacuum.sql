DO $$
DECLARE
  p regclass;
BEGIN
  FOR p IN
    SELECT relid
    FROM pg_partition_tree('marie_scheduler.job'::regclass)
    WHERE isleaf  -- only actual partitions
  LOOP
    EXECUTE format(
      'ALTER TABLE %s SET (
         autovacuum_vacuum_scale_factor = 0.02,
         autovacuum_analyze_scale_factor = 0.02
       )', p);
  END LOOP;
END$$;
