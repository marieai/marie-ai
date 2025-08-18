
-- TEMPLATE: generate DDL for all job partitions (returns rows of DDL text)
WITH parts AS (
  SELECT n.nspname AS schema_name, c.relname AS part_name, c.oid
  FROM pg_inherits i
  JOIN pg_class     c ON c.oid = i.inhrelid
  JOIN pg_class     p ON p.oid = i.inhparent
  JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE n.nspname = 'marie_scheduler'  -- << change schema if needed
    AND p.relname = 'job'              -- parent partitioned table
)
SELECT
  format(
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS %I ON %I.%I (name, state, job_level ASC, priority DESC) INCLUDE (id, dag_id) WHERE state IN (''created'',''retry'');',
    left(part_name, 35) || '_ready_ord_inc_' || substr(md5(oid::text), 1, 8),
    schema_name, part_name
  ) AS ddl
FROM parts



-- THIS NEEDS TO BE ADDED PER QUEUE (OUR INDEX ABOVE IS A GENERIC ONE)
) Corr/EXTRACT-filtered ordered (job_level, priority DESC, id) INCLUDE (dag_id) per partition
-- Copy the output and run the lines it prints (psql won’t wrap them in one transaction unless you use --single-transaction)
SELECT
  format(
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS %I ON %I.%I (job_level ASC, priority DESC, id) INCLUDE (dag_id) WHERE name = ''corr'' AND state IN (''created'',''retry'');',
    -- short, unique index name: <short_rel>_crr_sched_<8-char hash>
    left(c.relname, 35) || '_crr_sched_' || substr(md5(c.oid::text), 1, 8),
    n.nspname,
    c.relname
  ) AS ddl
FROM pg_inherits i
JOIN pg_class     c ON c.oid = i.inhrelid
JOIN pg_class     p ON p.oid = i.inhparent
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'marie_scheduler'
  AND p.relname = 'job'
ORDER BY c.relname;


-- used by fetch_next_job
 Generate the “not completed” partial index per partition (helps the dep check)
SELECT
  format(
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS %I ON %I.%I (id) WHERE state <> ''completed'';',
    left(c.relname, 35) || '_id_not_completed_' || substr(md5(c.oid::text), 1, 8),
    n.nspname,
    c.relname
  ) AS ddl
FROM pg_inherits i
JOIN pg_class     c ON c.oid = i.inhrelid
JOIN pg_class     p ON p.oid = i.inhparent
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'marie_scheduler'
  AND p.relname = 'job'
ORDER BY c.relname;


-- Per job partition (repeat on each child)
CREATE INDEX CONCURRENTLY IF NOT EXISTS <part>_id_idx ON marie_scheduler.<part> (id);

1) Plain (id) index per partition

SELECT
  format(
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS %I ON %I.%I (id);',
    left(c.relname, 35) || '_id_' || substr(md5(c.oid::text), 1, 8),
    n.nspname,
    c.relname
  ) AS ddl
FROM pg_inherits i
JOIN pg_class     c ON c.oid = i.inhrelid
JOIN pg_class     p ON p.oid = i.inhparent
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'marie_scheduler'
  AND p.relname = 'job'
ORDER BY c.relname;



2) Partial (id) WHERE state <> 'completed' per partition

SELECT
  format(
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS %I ON %I.%I (id) WHERE state <> ''completed'';',
    left(c.relname, 35) || '_id_not_completed_' || substr(md5(c.oid::text), 1, 8),
    n.nspname,
    c.relname
  ) AS ddl
FROM pg_inherits i
JOIN pg_class     c ON c.oid = i.inhrelid
JOIN pg_class     p ON p.oid = i.inhparent
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'marie_scheduler'
  AND p.relname = 'job'
ORDER BY c.relname;


 build the (dag_id, name) indexes on every marie_scheduler.job partition
 -- REDUCED from 1.5sec to ~350ms
-- USED BY count_dag_states
SELECT
  format(
    'CREATE INDEX IF NOT EXISTS %I ON %I.%I (dag_id, name);',
    left(c.relname, 35) || '_dag_name_' || substr(md5(c.oid::text), 1, 8),
    n.nspname,
    c.relname
  ) AS ddl
FROM pg_inherits i
JOIN pg_class     c ON c.oid = i.inhrelid
JOIN pg_class     p ON p.oid = i.inhparent
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'marie_scheduler'
  AND p.relname = 'job'
ORDER BY c.relname;


-- Analyze to update stats
ANALYZE marie_scheduler.job;
ANALYZE marie_scheduler.job_dependencies;
ANALYZE marie_scheduler.dag;

-- For hot job partitions you can also VACUUM to improve the visibility map:
VACUUM (ANALYZE) marie_scheduler.job_part_YYYYMM;


-- Per job partition (repeat on each child)
CREATE INDEX  job_corr_ready_idx
ON marie_scheduler.j9cd9b30b256e3ff2f0717d22abb91ab6b117fda746d660b7292735a6 (name, state, dag_id, id)
WHERE name = 'corr' AND state IN ('created','retry');

-- Per job partition (repeat on each child)
CREATE INDEX  job_extract_ready_idx
ON marie_scheduler.j5294dca0cf67eba9f6066f08560c47b010e0dce4a3ef60ff128d306e (name, state, dag_id, id)
WHERE name = 'extract' AND state IN ('created','retry');
