BEGIN;

CREATE TEMP TABLE scheduler_validation_lease_jobs ON COMMIT DROP AS
SELECT job.id
FROM marie_scheduler.job AS job
WHERE job.state IN ('created', 'retry')
  AND (
      job.lease_expires_at IS NULL
      OR job.lease_expires_at <= CURRENT_TIMESTAMP
  )
ORDER BY job.created_on, job.id
LIMIT 40;

ANALYZE scheduler_validation_lease_jobs;

SELECT count(*) AS available_test_jobs
FROM scheduler_validation_lease_jobs;

EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT unnest(
    marie_scheduler.lease_jobs_by_id(
        ARRAY(
            SELECT id
            FROM scheduler_validation_lease_jobs
            ORDER BY id
        ),
        INTERVAL '2 minutes',
        'scheduler-validation-lease',
        NULL
    )
);

ROLLBACK;

