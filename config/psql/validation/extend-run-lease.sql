BEGIN;

CREATE TEMP TABLE scheduler_validation_extend_job ON COMMIT DROP AS
SELECT
    job.id,
    job.run_owner,
    job.run_attempt_id
FROM marie_scheduler.job AS job
WHERE job.state = 'active'
  AND job.run_owner IS NOT NULL
  AND job.run_attempt_id IS NOT NULL
LIMIT 1;

UPDATE marie_scheduler.job AS job
SET run_lease_expires_at = CURRENT_TIMESTAMP + INTERVAL '1 minute'
FROM scheduler_validation_extend_job AS test_job
WHERE job.id = test_job.id;

ANALYZE scheduler_validation_extend_job;

SELECT count(*) AS available_test_jobs
FROM scheduler_validation_extend_job;

EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT unnest(
    marie_scheduler.extend_run_lease(
        ARRAY[test_job.id],
        test_job.run_owner,
        test_job.run_attempt_id,
        INTERVAL '5 minutes'
    )
)
FROM scheduler_validation_extend_job AS test_job;

ROLLBACK;

