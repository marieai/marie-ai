-- Read-only verification for the unpartitioned job cutover.
-- Run while scheduler writers are still stopped.

SELECT
    relation.relname,
    relation.relkind,
    pg_size_pretty(
        CASE WHEN relation.relkind = 'p' THEN (
            SELECT sum(pg_relation_size(tree.relid))
            FROM pg_partition_tree(relation.oid) AS tree
        ) ELSE pg_relation_size(relation.oid) END
    ) AS heap_size,
    pg_size_pretty(
        CASE WHEN relation.relkind = 'p' THEN (
            SELECT sum(pg_indexes_size(tree.relid))
            FROM pg_partition_tree(relation.oid) AS tree
        ) ELSE pg_indexes_size(relation.oid) END
    ) AS index_size,
    pg_size_pretty(
        CASE WHEN relation.relkind = 'p' THEN (
            SELECT sum(pg_total_relation_size(tree.relid))
            FROM pg_partition_tree(relation.oid) AS tree
        ) ELSE pg_total_relation_size(relation.oid) END
    ) AS total_size
FROM pg_class AS relation
WHERE relation.oid IN (
    'marie_scheduler.job'::regclass,
    'marie_scheduler.job_partitioned_old'::regclass
)
ORDER BY relation.relname;

SELECT
    count(*) AS active_partitions
FROM pg_inherits
WHERE inhparent = 'marie_scheduler.job'::regclass;

SELECT
    (SELECT count(*) FROM marie_scheduler.job) AS active_rows,
    (SELECT count(*) FROM marie_scheduler.job_partitioned_old) AS rollback_rows;

SELECT
    active.state,
    active.row_count AS active_rows,
    retained.row_count AS rollback_rows,
    active.row_count = retained.row_count AS matches
FROM (
    SELECT state::text AS state, count(*) AS row_count
    FROM marie_scheduler.job
    GROUP BY state::text
) AS active
FULL JOIN (
    SELECT state::text AS state, count(*) AS row_count
    FROM marie_scheduler.job_partitioned_old
    GROUP BY state::text
) AS retained USING (state)
ORDER BY state;

EXPLAIN (ANALYZE, BUFFERS, WAL, SETTINGS, VERBOSE, SUMMARY)
WITH sample_dags AS MATERIALIZED (
    SELECT job.dag_id
    FROM marie_scheduler.job AS job
    WHERE job.state IN ('created', 'retry')
    GROUP BY job.dag_id
    ORDER BY count(*) DESC
    LIMIT 100
)
SELECT hydrated.dag_id, hydrated.job
FROM marie_scheduler.hydrate_frontier_jobs(
    ARRAY(SELECT sample_dags.dag_id FROM sample_dags)
) AS hydrated;

EXPLAIN (ANALYZE, BUFFERS, WAL, SETTINGS, VERBOSE, SUMMARY)
SELECT candidate.dag_id, candidate.serialized_dag
FROM marie_scheduler.admission_candidate_dags(
    100,
    600,
    ARRAY[]::uuid[]
) WITH ORDINALITY AS candidate(
    dag_id,
    serialized_dag,
    admission_rank
)
ORDER BY candidate.admission_rank;
