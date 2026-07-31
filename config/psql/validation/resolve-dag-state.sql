BEGIN;

CREATE TEMP TABLE scheduler_validation_resolve_dag ON COMMIT DROP AS
SELECT job.dag_id, count(*) AS job_count
FROM marie_scheduler.job AS job
GROUP BY job.dag_id
ORDER BY count(*) DESC, job.dag_id
LIMIT 1;

ANALYZE scheduler_validation_resolve_dag;

SELECT dag_id, job_count
FROM scheduler_validation_resolve_dag;

EXPLAIN (
    ANALYZE,
    BUFFERS,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT EXISTS (
    SELECT 1
    FROM marie_scheduler.job AS job
    WHERE job.dag_id = (
        SELECT dag_id
        FROM scheduler_validation_resolve_dag
    )
      AND job.state::text IN ('failed', 'expired', 'cancelled')
);

EXPLAIN (
    ANALYZE,
    BUFFERS,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT NOT EXISTS (
    SELECT 1
    FROM marie_scheduler.job AS job
    WHERE job.dag_id = (
        SELECT dag_id
        FROM scheduler_validation_resolve_dag
    )
      AND job.state::text NOT IN ('completed', 'skipped')
);

EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT marie_scheduler.resolve_dag_state(test_dag.dag_id)
FROM scheduler_validation_resolve_dag AS test_dag;

ROLLBACK;

