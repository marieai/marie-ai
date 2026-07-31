BEGIN;

SELECT count(*) AS releasable_leases
FROM (
    SELECT job.id
    FROM marie_scheduler.job AS job
    WHERE job.state IN ('created', 'retry')
      AND job.lease_owner IS NOT NULL
      AND job.lease_expires_at IS NOT NULL
      AND job.lease_expires_at <= CURRENT_TIMESTAMP
    ORDER BY job.lease_expires_at, job.id
    LIMIT 1000
) AS expired;

EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT marie_scheduler.release_expired_leases(1000);

ROLLBACK;

