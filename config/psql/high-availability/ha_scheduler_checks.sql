-- Marie Scheduler High Availability Checks
--
-- This file is plain PostgreSQL SQL. It is safe to run from database clients
-- such as DataGrip, PyCharm Database, DBeaver, psql, or the Postgres console.
--
-- Edit the ha_check_params statement below to scope the run. The default scans
-- the last 24 hours.
--
-- The script only creates temporary tables. It does not modify persistent
-- scheduler tables.

-- Setup: define the run window used by every check in this file.
DROP TABLE IF EXISTS ha_check_params;

CREATE TEMP TABLE ha_check_params AS
SELECT
    now() - INTERVAL '24 hours' AS run_start,
    now() AS run_end;

-- For an exact HA run window, replace the CREATE TEMP TABLE above with:
--
-- CREATE TEMP TABLE ha_check_params AS
-- SELECT
--     '2026-05-19 09:45:00+00'::timestamptz AS run_start,
--     '2026-05-19 10:00:00+00'::timestamptz AS run_end;

-- Query: show the effective run window so copied results are self-describing.
SELECT
    'marie_scheduler' AS schema_name,
    run_start,
    run_end
FROM ha_check_params;

-- Setup: materialize the attempt rows for the chosen window so later queries
-- all inspect the same snapshot.
DROP TABLE IF EXISTS ha_scoped_attempts;

CREATE TEMP TABLE ha_scoped_attempts AS
SELECT *
FROM marie_scheduler.job_attempt
WHERE activated_at >= (SELECT run_start FROM ha_check_params)
  AND activated_at < (SELECT run_end FROM ha_check_params);

-- Query: count how many job attempts are included in this HA check.
SELECT COUNT(*) AS scoped_attempts
FROM ha_scoped_attempts;

-- Query: show which gateway/scheduler pair activated, dispatched, completed,
-- or recovered attempts. This is the primary "who processed what" view.
SELECT
    gateway_instance_id,
    scheduler_lease_owner,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE dispatch_started_at IS NOT NULL) AS dispatch_started,
    COUNT(*) FILTER (WHERE dispatch_confirmed_at IS NOT NULL) AS dispatched,
    COUNT(*) FILTER (WHERE terminal_accepted IS TRUE) AS terminal_accepted,
    COUNT(*) FILTER (WHERE recovery_state IS NOT NULL) AS recovered,
    COUNT(*) FILTER (
        WHERE dispatch_confirmed_at IS NOT NULL
          AND terminal_accepted IS DISTINCT FROM TRUE
          AND recovery_state IS NULL
    ) AS dispatched_missing_terminal
FROM ha_scoped_attempts
GROUP BY gateway_instance_id, scheduler_lease_owner
ORDER BY attempts DESC, gateway_instance_id;

-- Query: summarize terminal event handling by source and rejection reason.
-- Normal no-fault runs should mostly show accepted job_event terminals.
SELECT
    terminal_accepted,
    terminal_source,
    terminal_reject_reason,
    COUNT(*) AS count
FROM ha_scoped_attempts
GROUP BY terminal_accepted, terminal_source, terminal_reject_reason
ORDER BY count DESC, terminal_accepted NULLS FIRST, terminal_source;

-- Query: show whether terminal events were accepted by the same gateway that
-- activated the attempt or by a different gateway during HA handoff.
SELECT
    gateway_instance_id AS activated_gateway,
    terminal_gateway_instance_id AS terminal_gateway,
    COUNT(*) AS terminal_events,
    COUNT(*) FILTER (WHERE terminal_accepted IS TRUE) AS terminal_accepted,
    COUNT(*) FILTER (WHERE terminal_accepted IS FALSE) AS terminal_rejected
FROM ha_scoped_attempts
WHERE terminal_at IS NOT NULL
GROUP BY gateway_instance_id, terminal_gateway_instance_id
ORDER BY terminal_events DESC, activated_gateway, terminal_gateway;

-- Query: summarize attempt lifecycle states across activation, dispatch,
-- terminal handling, and recovery.
SELECT
    attempt_state,
    terminal_status,
    terminal_work_state,
    terminal_source,
    terminal_accepted,
    recovery_state,
    COUNT(*) AS count
FROM ha_scoped_attempts
GROUP BY
    attempt_state,
    terminal_status,
    terminal_work_state,
    terminal_source,
    terminal_accepted,
    recovery_state
ORDER BY count DESC;

-- Query: compare attempt audit state against the current authoritative job
-- state in the scheduler table.
SELECT
    j.state::text AS job_state,
    COUNT(DISTINCT a.job_id) AS jobs,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE a.dispatch_confirmed_at IS NOT NULL) AS dispatched_attempts,
    COUNT(*) FILTER (WHERE a.terminal_accepted IS TRUE) AS terminal_accepted_attempts,
    COUNT(*) FILTER (WHERE a.recovery_state IS NOT NULL) AS recovered_attempts
FROM ha_scoped_attempts a
JOIN marie_scheduler.job j ON j.id = a.job_id
GROUP BY j.state::text
ORDER BY jobs DESC, job_state;

-- Query: machine-readable PASS/FAIL summary for HA correctness. Investigate
-- any row where result = FAIL, unless the failure is expected by a fault test.
WITH checks AS (
    SELECT
        'active_active_gateway_count' AS check_name,
        COUNT(DISTINCT gateway_instance_id) FILTER (
            WHERE dispatch_confirmed_at IS NOT NULL
              AND gateway_instance_id IS NOT NULL
        ) AS observed_count,
        CASE
            WHEN COUNT(DISTINCT gateway_instance_id) FILTER (
                WHERE dispatch_confirmed_at IS NOT NULL
                  AND gateway_instance_id IS NOT NULL
            ) >= 2 THEN 0
            ELSE 1
        END AS bad_rows,
        'At least two gateway instances dispatched work during an active-active HA test.' AS expectation
    FROM ha_scoped_attempts

    UNION ALL
    SELECT
        'dispatched_missing_terminal',
        COUNT(*),
        COUNT(*),
        'After drain, every dispatched attempt should have an accepted terminal event or a recovery marker.'
    FROM ha_scoped_attempts
    WHERE dispatch_confirmed_at IS NOT NULL
      AND terminal_accepted IS DISTINCT FROM TRUE
      AND recovery_state IS NULL

    UNION ALL
    SELECT
        'completed_dispatched_missing_terminal',
        COUNT(*),
        COUNT(*),
        'Completed terminal jobs with confirmed dispatch must have terminal audit.'
    FROM ha_scoped_attempts a
    JOIN marie_scheduler.job j ON j.id = a.job_id
    WHERE j.state::text IN ('completed', 'failed', 'cancelled', 'expired', 'skipped')
      AND a.dispatch_confirmed_at IS NOT NULL
      AND a.terminal_accepted IS DISTINCT FROM TRUE
      AND a.recovery_state IS NULL

    UNION ALL
    SELECT
        'terminal_rejected',
        COUNT(*),
        COUNT(*),
        'Normal no-fault HA runs should not reject terminal events. Fault tests may intentionally reject stale attempts.'
    FROM ha_scoped_attempts
    WHERE terminal_accepted IS FALSE

    UNION ALL
    SELECT
        'duplicate_completed_terminal_by_job',
        COUNT(*),
        COUNT(*),
        'A job should not have more than one accepted completed terminal attempt.'
    FROM (
        SELECT job_id
        FROM ha_scoped_attempts
        WHERE terminal_accepted IS TRUE
          AND terminal_work_state = 'completed'
        GROUP BY job_id
        HAVING COUNT(*) > 1
    ) duplicate_completed

    UNION ALL
    SELECT
        'dispatched_without_gateway_instance',
        COUNT(*),
        COUNT(*),
        'Every dispatched attempt should record the gateway instance that activated it.'
    FROM ha_scoped_attempts
    WHERE dispatch_confirmed_at IS NOT NULL
      AND gateway_instance_id IS NULL

    UNION ALL
    SELECT
        'terminal_without_gateway_instance',
        COUNT(*),
        COUNT(*),
        'Every accepted terminal event should record the gateway instance that handled it.'
    FROM ha_scoped_attempts
    WHERE terminal_accepted IS TRUE
      AND terminal_gateway_instance_id IS NULL

    UNION ALL
    SELECT
        'active_missing_attempt_identity',
        COUNT(*),
        COUNT(*),
        'Active jobs in the run window must have run_owner, run_attempt_id, and run_lease_expires_at.'
    FROM marie_scheduler.job j
    WHERE COALESCE(j.started_on, j.created_on) >= (SELECT run_start FROM ha_check_params)
      AND COALESCE(j.started_on, j.created_on) < (SELECT run_end FROM ha_check_params)
      AND j.state::text = 'active'
      AND (
          j.run_owner IS NULL
          OR j.run_attempt_id IS NULL
          OR j.run_lease_expires_at IS NULL
      )

    UNION ALL
    SELECT
        'expired_active_run_leases',
        COUNT(*),
        COUNT(*),
        'After maintenance settles, active jobs in the run window should not have expired run leases.'
    FROM marie_scheduler.job j
    WHERE COALESCE(j.started_on, j.created_on) >= (SELECT run_start FROM ha_check_params)
      AND COALESCE(j.started_on, j.created_on) < (SELECT run_end FROM ha_check_params)
      AND j.state::text = 'active'
      AND j.run_lease_expires_at < now()
)
-- Query: return the HA correctness check results. Rows with result = FAIL need
-- investigation unless the failure is expected by a fault-injection test.
SELECT
    check_name,
    observed_count,
    bad_rows,
    CASE WHEN bad_rows = 0 THEN 'PASS' ELSE 'FAIL' END AS result,
    expectation
FROM checks
ORDER BY
    CASE WHEN bad_rows = 0 THEN 1 ELSE 0 END,
    check_name;

-- Query: detail rows for confirmed dispatches that still have no accepted
-- terminal audit and no recovery marker. This should be empty after drain.
SELECT
    a.job_id,
    a.run_attempt_id,
    j.state::text AS job_state,
    j.completed_on,
    a.gateway_instance_id,
    a.scheduler_lease_owner,
    a.executor,
    a.dispatch_started_at,
    a.dispatch_confirmed_at,
    a.attempt_state,
    a.terminal_at,
    a.terminal_accepted,
    a.terminal_reject_reason,
    a.recovery_state,
    a.dispatch_error
FROM ha_scoped_attempts a
JOIN marie_scheduler.job j ON j.id = a.job_id
WHERE a.dispatch_confirmed_at IS NOT NULL
  AND a.terminal_accepted IS DISTINCT FROM TRUE
  AND a.recovery_state IS NULL
ORDER BY a.dispatch_confirmed_at DESC NULLS LAST
LIMIT 50;

-- Query: detail rows for jobs that accepted more than one completed terminal
-- attempt. This should be empty; non-empty rows indicate duplicate completion.
SELECT
    a.job_id,
    COUNT(*) AS accepted_completed_attempts,
    array_agg(a.run_attempt_id ORDER BY a.terminal_at) AS run_attempt_ids,
    min(a.terminal_at) AS first_terminal_at,
    max(a.terminal_at) AS last_terminal_at
FROM ha_scoped_attempts a
WHERE a.terminal_accepted IS TRUE
  AND a.terminal_work_state = 'completed'
GROUP BY a.job_id
HAVING COUNT(*) > 1
ORDER BY accepted_completed_attempts DESC, last_terminal_at DESC
LIMIT 50;

-- Query: detail rows for rejected terminal events. In normal no-fault runs this
-- should be empty; in stale-attempt tests, rejection rows should be expected.
SELECT
    a.job_id,
    a.run_attempt_id,
    j.state::text AS job_state,
    a.gateway_instance_id AS activated_gateway,
    a.terminal_gateway_instance_id AS terminal_gateway,
    a.terminal_at,
    a.terminal_status,
    a.terminal_work_state,
    a.terminal_source,
    a.terminal_reject_reason
FROM ha_scoped_attempts a
JOIN marie_scheduler.job j ON j.id = a.job_id
WHERE a.terminal_accepted IS FALSE
ORDER BY a.terminal_at DESC NULLS LAST
LIMIT 50;

-- Query: detail rows for active jobs whose run lease has expired. This should
-- be empty after maintenance/recovery has had time to run.
SELECT
    j.id,
    j.name,
    j.state::text AS job_state,
    j.run_owner,
    j.run_attempt_id,
    j.lease_owner,
    j.lease_expires_at,
    j.run_lease_expires_at,
    j.started_on,
    j.created_on
FROM marie_scheduler.job j
WHERE COALESCE(j.started_on, j.created_on) >= (SELECT run_start FROM ha_check_params)
  AND COALESCE(j.started_on, j.created_on) < (SELECT run_end FROM ha_check_params)
  AND j.state::text = 'active'
  AND j.run_lease_expires_at < now()
ORDER BY j.run_lease_expires_at
LIMIT 50;

-- Query: show jobs with multiple attempts in the run window. Retries and
-- recovery can make this valid, but it is useful context for HA debugging.
SELECT
    a.job_id,
    j.state::text AS job_state,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE a.dispatch_confirmed_at IS NOT NULL) AS dispatched,
    COUNT(*) FILTER (WHERE a.terminal_accepted IS TRUE) AS terminal_accepted,
    COUNT(*) FILTER (WHERE a.recovery_state IS NOT NULL) AS recovered,
    array_agg(a.run_attempt_id ORDER BY a.activated_at) AS run_attempt_ids
FROM ha_scoped_attempts a
JOIN marie_scheduler.job j ON j.id = a.job_id
GROUP BY a.job_id, j.state::text
HAVING COUNT(*) > 1
ORDER BY attempts DESC, a.job_id
LIMIT 50;
