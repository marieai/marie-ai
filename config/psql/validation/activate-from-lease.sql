BEGIN;

CREATE TEMP TABLE scheduler_validation_activate_candidates ON COMMIT DROP AS
SELECT job.id
FROM marie_scheduler.job AS job
WHERE job.state IN ('created', 'retry')
  AND (
      job.lease_expires_at IS NULL
      OR job.lease_expires_at <= CURRENT_TIMESTAMP
  )
ORDER BY job.created_on, job.id
LIMIT 40;

CREATE TEMP TABLE scheduler_validation_activate_jobs ON COMMIT DROP AS
SELECT unnest(
    marie_scheduler.lease_jobs_by_id(
        ARRAY(
            SELECT id
            FROM scheduler_validation_activate_candidates
            ORDER BY id
        ),
        INTERVAL '2 minutes',
        'scheduler-validation-activate',
        NULL
    )
) AS id;

ANALYZE scheduler_validation_activate_jobs;

SELECT count(*) AS leased_test_jobs
FROM scheduler_validation_activate_jobs;

EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT activated.job_id, activated.run_attempt_id
FROM marie_scheduler.activate_from_lease(
    ARRAY(
        SELECT id
        FROM scheduler_validation_activate_jobs
        ORDER BY id
    ),
    'scheduler-validation-activate',
    INTERVAL '5 minutes',
    'scheduler-validation'
) AS activated;

ROLLBACK;

