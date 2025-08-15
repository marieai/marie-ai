

3) Corr-filtered ordered (job_level, priority DESC, id) INCLUDE (dag_id) per partition
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

"CREATE INDEX CONCURRENTLY IF NOT EXISTS j31a837722a37afbac94ad1e6be5a9c8478_crr_sched_f138e593 ON marie_scheduler.j31a837722a37afbac94ad1e6be5a9c84789ac02630bd52dd76129b59 (job_level ASC, priority DESC, id) INCLUDE (dag_id) WHERE name = 'corr' AND state IN ('created','retry');"
"CREATE INDEX CONCURRENTLY IF NOT EXISTS j5294dca0cf67eba9f6066f08560c47b010_crr_sched_dd96eaef ON marie_scheduler.j5294dca0cf67eba9f6066f08560c47b010e0dce4a3ef60ff128d306e (job_level ASC, priority DESC, id) INCLUDE (dag_id) WHERE name = 'corr' AND state IN ('created','retry');"
"CREATE INDEX CONCURRENTLY IF NOT EXISTS j725a7a526290f881194191b8d3ef7129f3_crr_sched_fa7fb9f1 ON marie_scheduler.j725a7a526290f881194191b8d3ef7129f3df548560ceec816f9f69be (job_level ASC, priority DESC, id) INCLUDE (dag_id) WHERE name = 'corr' AND state IN ('created','retry');"
"CREATE INDEX CONCURRENTLY IF NOT EXISTS j9cd9b30b256e3ff2f0717d22abb91ab6b1_crr_sched_39e947eb ON marie_scheduler.j9cd9b30b256e3ff2f0717d22abb91ab6b117fda746d660b7292735a6 (job_level ASC, priority DESC, id) INCLUDE (dag_id) WHERE name = 'corr' AND state IN ('created','retry');"
"CREATE INDEX CONCURRENTLY IF NOT EXISTS jdd633563ce11415bf7bbdd408e1afb0a97_crr_sched_bd44f20e ON marie_scheduler.jdd633563ce11415bf7bbdd408e1afb0a97887270bab5b26f6c8d8f49 (job_level ASC, priority DESC, id) INCLUDE (dag_id) WHERE name = 'corr' AND state IN ('created','retry');"
"CREATE INDEX CONCURRENTLY IF NOT EXISTS jef11aecfd8914e8e5437ebd9b167af245f_crr_sched_32e0740f ON marie_scheduler.jef11aecfd8914e8e5437ebd9b167af245fdd0a4332f0056c892f40fb (job_level ASC, priority DESC, id) INCLUDE (dag_id) WHERE name = 'corr' AND state IN ('created','retry');"



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

CREATE INDEX CONCURRENTLY IF NOT EXISTS j31a837722a37afbac94ad1e6be5a9c8478_id_not_completed_f138e593 ON marie_scheduler.j31a837722a37afbac94ad1e6be5a9c84789ac02630bd52dd76129b59 (id) WHERE state <> 'completed';
CREATE INDEX CONCURRENTLY IF NOT EXISTS j5294dca0cf67eba9f6066f08560c47b010_id_not_completed_dd96eaef ON marie_scheduler.j5294dca0cf67eba9f6066f08560c47b010e0dce4a3ef60ff128d306e (id) WHERE state <> 'completed';
CREATE INDEX CONCURRENTLY IF NOT EXISTS j725a7a526290f881194191b8d3ef7129f3_id_not_completed_fa7fb9f1 ON marie_scheduler.j725a7a526290f881194191b8d3ef7129f3df548560ceec816f9f69be (id) WHERE state <> 'completed';
CREATE INDEX CONCURRENTLY IF NOT EXISTS j9cd9b30b256e3ff2f0717d22abb91ab6b1_id_not_completed_39e947eb ON marie_scheduler.j9cd9b30b256e3ff2f0717d22abb91ab6b117fda746d660b7292735a6 (id) WHERE state <> 'completed';
CREATE INDEX CONCURRENTLY IF NOT EXISTS jdd633563ce11415bf7bbdd408e1afb0a97_id_not_completed_bd44f20e ON marie_scheduler.jdd633563ce11415bf7bbdd408e1afb0a97887270bab5b26f6c8d8f49 (id) WHERE state <> 'completed';
CREATE INDEX CONCURRENTLY IF NOT EXISTS jef11aecfd8914e8e5437ebd9b167af245f_id_not_completed_32e0740f ON marie_scheduler.jef11aecfd8914e8e5437ebd9b167af245fdd0a4332f0056c892f40fb (id) WHERE state <> 'completed';




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
