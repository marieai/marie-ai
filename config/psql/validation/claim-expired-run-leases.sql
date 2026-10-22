BEGIN;

EXPLAIN (
    ANALYZE,
    BUFFERS,
    WAL,
    SETTINGS,
    VERBOSE,
    SUMMARY
)
SELECT *
FROM marie_scheduler.claim_expired_run_leases(NULL::text[], 1000);

ROLLBACK;

