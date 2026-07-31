BEGIN;

CREATE TEMP TABLE scheduler_validation_hydrate_dags ON COMMIT DROP AS
SELECT job.dag_id
FROM marie_scheduler.job AS job
WHERE job.state IN ('created', 'retry')
GROUP BY job.dag_id
ORDER BY count(*) DESC, job.dag_id
LIMIT 100;

ANALYZE scheduler_validation_hydrate_dags;

SELECT count(*) AS tested_dags
FROM scheduler_validation_hydrate_dags;

EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT hydrated.dag_id, hydrated.job
FROM marie_scheduler.hydrate_frontier_jobs(
    ARRAY(
        SELECT dag_id
        FROM scheduler_validation_hydrate_dags
        ORDER BY dag_id
    )
) AS hydrated;

ROLLBACK;

