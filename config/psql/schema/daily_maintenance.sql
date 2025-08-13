
-- =========================================================
-- daily maintenance
-- =========================================================
CREATE OR REPLACE FUNCTION marie_scheduler.daily_maintenance() RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
  moved_jobs bigint;
  moved_dags bigint;
  reindexed int;
BEGIN
  PERFORM set_config('maintenance_work_mem','2GB', false);

  PERFORM marie_scheduler.purge_history('14 days','14 days', 20000);

  reindexed := marie_scheduler.reindex_hot_indexes();
  RAISE NOTICE 'reindexed % hot indexes', reindexed;

  -- Analyze updated tables
  ANALYZE dag;
  ANALYZE job;
  ANALYZE job_dependencies;
  ANALYZE job_history;
  ANALYZE dag_history;
END$$;

