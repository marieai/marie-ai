-- Marie HA In-Flight Gateway Kill Invariants
--
-- This file is plain PostgreSQL SQL. It is safe to run from database clients
-- such as DataGrip, PyCharm Database, DBeaver, psql, or the Postgres console.
--
-- Use this after a test where:
--   1. two gateways are running,
--   2. a long-running job is active,
--   3. the gateway owning that active attempt is killed,
--   4. the system is allowed to settle for run_ttl_seconds + maintenance buffer.
--
-- The script only creates temporary tables. It does not modify persistent
-- scheduler tables.

-- Setup: define the test window and, optionally, the killed gateway/owner from
-- current-session marie.ha_* settings. Unset values retain safe defaults.
DROP TABLE IF EXISTS pg_temp.ha_kill_check_params;

CREATE TEMP TABLE ha_kill_check_params AS
SELECT
    COALESCE(
        NULLIF(current_setting('marie.ha_run_start', TRUE), '')::timestamptz,
        now() - INTERVAL '2 hours'
    ) AS run_start,
    COALESCE(
        NULLIF(current_setting('marie.ha_run_end', TRUE), '')::timestamptz,
        now()
    ) AS run_end,
    NULLIF(
        current_setting('marie.ha_killed_gateway_instance_id', TRUE), ''
    )::text AS killed_gateway_instance_id,
    NULLIF(
        current_setting('marie.ha_killed_scheduler_lease_owner', TRUE), ''
    )::text AS killed_scheduler_lease_owner;

-- Example exact window with known killed gateway:
--
-- SET marie.ha_run_start = '2026-05-19 10:00:00+00';
-- SET marie.ha_run_end = '2026-05-19 10:10:00+00';
-- SET marie.ha_killed_gateway_instance_id = 'xpredator:gateway-instance-id';
-- SET marie.ha_killed_scheduler_lease_owner = 'xpredator:scheduler-lease-owner';

-- Query: show the effective test window and optional killed owner filters.
SELECT
    run_start,
    run_end,
    killed_gateway_instance_id,
    killed_scheduler_lease_owner
FROM ha_kill_check_params;

-- Setup: materialize the attempt rows for the selected test window.
DROP TABLE IF EXISTS pg_temp.ha_kill_attempts;

CREATE TEMP TABLE ha_kill_attempts AS
SELECT *
FROM marie_scheduler.job_attempt
WHERE activated_at >= (SELECT run_start FROM ha_kill_check_params)
  AND activated_at < (SELECT run_end FROM ha_kill_check_params);

-- Query: if run before killing a gateway, this shows active attempts and their
-- owning gateway/scheduler. Use these values to decide which process to kill.
SELECT
    j.id AS job_id,
    j.name,
    j.state::text AS job_state,
    j.run_owner,
    j.run_attempt_id,
    j.run_lease_expires_at,
    ja.gateway_instance_id,
    ja.scheduler_lease_owner,
    ja.dispatch_started_at,
    ja.dispatch_confirmed_at
FROM marie_scheduler.job j
JOIN marie_scheduler.job_attempt ja
  ON ja.run_attempt_id = j.run_attempt_id
WHERE j.state::text = 'active'
ORDER BY j.started_on DESC NULLS LAST
LIMIT 25;

-- Query: summarize attempts by gateway/scheduler for the test window. After a
-- kill test, this should show the killed gateway's dispatched work and the
-- surviving gateway's later completions/recoveries, depending on the scenario.
SELECT
    gateway_instance_id,
    scheduler_lease_owner,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE dispatch_confirmed_at IS NOT NULL) AS dispatched,
    COUNT(*) FILTER (WHERE terminal_accepted IS TRUE) AS terminal_accepted,
    COUNT(*) FILTER (WHERE terminal_accepted IS FALSE) AS terminal_rejected,
    COUNT(*) FILTER (WHERE recovery_state IS NOT NULL) AS recovered,
    COUNT(*) FILTER (
        WHERE dispatch_confirmed_at IS NOT NULL
          AND terminal_accepted IS DISTINCT FROM TRUE
          AND recovery_state IS NULL
    ) AS dispatched_missing_terminal_or_recovery
FROM ha_kill_attempts
GROUP BY gateway_instance_id, scheduler_lease_owner
ORDER BY attempts DESC, gateway_instance_id;

-- Query: summarize rows for the killed gateway/owner when the optional filters
-- are filled in. If both filters are NULL, this returns zero rows.
SELECT
    gateway_instance_id,
    scheduler_lease_owner,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE dispatch_confirmed_at IS NOT NULL) AS dispatched,
    COUNT(*) FILTER (WHERE terminal_accepted IS TRUE) AS terminal_accepted,
    COUNT(*) FILTER (WHERE terminal_accepted IS FALSE) AS terminal_rejected,
    COUNT(*) FILTER (WHERE recovery_state IS NOT NULL) AS recovered
FROM ha_kill_attempts
WHERE (
        (SELECT killed_gateway_instance_id FROM ha_kill_check_params) IS NOT NULL
        AND gateway_instance_id = (
            SELECT killed_gateway_instance_id FROM ha_kill_check_params
        )
    )
   OR (
        (SELECT killed_scheduler_lease_owner FROM ha_kill_check_params) IS NOT NULL
        AND scheduler_lease_owner = (
            SELECT killed_scheduler_lease_owner FROM ha_kill_check_params
        )
    )
GROUP BY gateway_instance_id, scheduler_lease_owner
ORDER BY attempts DESC, gateway_instance_id;

-- Query: PASS/FAIL invariant summary. Common attempt, lease, and terminal
-- semantics come from the same schema function used by the scheduler stress
-- verifier. The recovery check below is specific to the kill scenario.
WITH shared_checks AS (
    SELECT check_name, bad_rows, expectation
    FROM marie_scheduler.scheduler_attempt_invariant_checks(
        NULL,
        (SELECT run_start FROM ha_kill_check_params),
        (SELECT run_end FROM ha_kill_check_params),
        (SELECT run_end FROM ha_kill_check_params),
        50
    )
), kill_specific_checks AS (
    SELECT
        'recovered_attempt_still_expired_active' AS check_name,
        COUNT(*) AS bad_rows,
        'Recovered attempts must not leave the job in an expired active state.'
            AS expectation
    FROM ha_kill_attempts a
    JOIN marie_scheduler.job j ON j.id = a.job_id
    WHERE a.recovery_state IS NOT NULL
      AND j.state::text = 'active'
      AND j.run_lease_expires_at < now()
), checks AS (
    SELECT check_name, bad_rows, expectation FROM shared_checks
    UNION ALL
    SELECT check_name, bad_rows, expectation FROM kill_specific_checks
)
SELECT
    check_name,
    bad_rows,
    CASE WHEN bad_rows = 0 THEN 'PASS' ELSE 'FAIL' END AS result,
    expectation
FROM checks
ORDER BY
    CASE WHEN bad_rows = 0 THEN 1 ELSE 0 END,
    check_name;

-- Query: detail rows for active jobs missing required attempt identity. This
-- should be empty.
SELECT
    id,
    name,
    state::text AS job_state,
    run_owner,
    run_attempt_id,
    run_lease_expires_at,
    started_on,
    created_on
FROM marie_scheduler.job
WHERE COALESCE(started_on, created_on) >= (
        SELECT run_start FROM ha_kill_check_params
    )
  AND COALESCE(started_on, created_on) < (
        SELECT run_end FROM ha_kill_check_params
    )
  AND state::text = 'active'
  AND (
      run_owner IS NULL
      OR run_attempt_id IS NULL
      OR run_lease_expires_at IS NULL
  )
ORDER BY started_on DESC NULLS LAST
LIMIT 50;

-- Query: detail rows for active jobs whose run lease has expired. This should
-- be empty after maintenance/recovery has had time to run.
SELECT
    id,
    name,
    state::text AS job_state,
    run_owner,
    run_attempt_id,
    run_lease_expires_at,
    now() AS checked_at
FROM marie_scheduler.job
WHERE COALESCE(started_on, created_on) >= (
        SELECT run_start FROM ha_kill_check_params
    )
  AND COALESCE(started_on, created_on) < (
        SELECT run_end FROM ha_kill_check_params
    )
  AND state::text = 'active'
  AND run_lease_expires_at < now()
ORDER BY run_lease_expires_at
LIMIT 50;

-- Query: detail rows for dispatched attempts that have neither terminal audit
-- nor recovery audit. This should be empty after the workload drains.
SELECT
    a.job_id,
    a.run_attempt_id,
    j.state::text AS current_job_state,
    a.gateway_instance_id,
    a.scheduler_lease_owner,
    a.dispatch_started_at,
    a.dispatch_confirmed_at,
    a.terminal_accepted,
    a.terminal_reject_reason,
    a.recovery_state,
    a.recovery_reason
FROM ha_kill_attempts a
JOIN marie_scheduler.job j ON j.id = a.job_id
WHERE a.dispatch_confirmed_at IS NOT NULL
  AND a.terminal_accepted IS DISTINCT FROM TRUE
  AND a.recovery_state IS NULL
ORDER BY a.dispatch_confirmed_at DESC NULLS LAST
LIMIT 50;

-- Query: detail rows for duplicate accepted completed terminals. This should
-- be empty. Non-empty rows indicate duplicate completion was accepted.
SELECT
    job_id,
    COUNT(*) AS accepted_completed_attempts,
    array_agg(run_attempt_id ORDER BY terminal_at) AS run_attempt_ids,
    min(terminal_at) AS first_terminal_at,
    max(terminal_at) AS last_terminal_at
FROM ha_kill_attempts
WHERE terminal_accepted IS TRUE
  AND terminal_work_state = 'completed'
GROUP BY job_id
HAVING COUNT(*) > 1
ORDER BY accepted_completed_attempts DESC, last_terminal_at DESC
LIMIT 50;

-- Query: show recovered jobs and their current durable state. Recovered jobs
-- should be terminal or running a newer valid active attempt, not expired-active.
SELECT
    a.job_id,
    j.state::text AS current_job_state,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE a.recovery_state IS NOT NULL) AS recovered_attempts,
    COUNT(*) FILTER (WHERE a.terminal_accepted IS TRUE) AS accepted_terminals,
    min(a.activated_at) AS first_activated_at,
    max(a.activated_at) AS last_activated_at,
    array_agg(a.run_attempt_id ORDER BY a.activated_at) AS attempts_seen
FROM ha_kill_attempts a
JOIN marie_scheduler.job j ON j.id = a.job_id
GROUP BY a.job_id, j.state::text
HAVING COUNT(*) FILTER (WHERE a.recovery_state IS NOT NULL) > 0
ORDER BY recovered_attempts DESC, attempts DESC, a.job_id
LIMIT 100;

-- Query: inspect terminal rejections. These may be expected if the killed
-- gateway's old attempt reports after recovery. The reject reason should show
-- stale owner/attempt behavior, not a missing audit or schema problem.
SELECT
    job_id,
    run_attempt_id,
    gateway_instance_id AS activated_gateway,
    terminal_gateway_instance_id,
    terminal_at,
    terminal_status,
    terminal_work_state,
    terminal_source,
    terminal_reject_reason
FROM ha_kill_attempts
WHERE terminal_accepted IS FALSE
ORDER BY terminal_at DESC NULLS LAST
LIMIT 100;

-- Query: show jobs with multiple attempts. Multiple attempts are valid when
-- recovery or retry occurs; this is context for reviewing the kill test.
SELECT
    a.job_id,
    j.state::text AS current_job_state,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE a.dispatch_confirmed_at IS NOT NULL) AS dispatched,
    COUNT(*) FILTER (WHERE a.terminal_accepted IS TRUE) AS accepted_terminals,
    COUNT(*) FILTER (WHERE a.terminal_accepted IS FALSE) AS rejected_terminals,
    COUNT(*) FILTER (WHERE a.recovery_state IS NOT NULL) AS recovered,
    array_agg(a.run_attempt_id ORDER BY a.activated_at) AS attempts_seen
FROM ha_kill_attempts a
JOIN marie_scheduler.job j ON j.id = a.job_id
GROUP BY a.job_id, j.state::text
HAVING COUNT(*) > 1
ORDER BY attempts DESC, a.job_id
LIMIT 100;

-- Note: semaphore leaks are not a PostgreSQL invariant because scheduler
-- semaphores live in etcd. Pair this DB script with an etcd/SemaphoreStore
-- holder/count check for the executor slot type used by the test.
