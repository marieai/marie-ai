-- Marie HA Lost Local Terminal Event Reconciliation Test
--
-- Purpose:
--   Validate the safety-net path where durable job-info storage says a job is
--   terminal, but the gateway-local scheduler event was lost. The expected
--   reconciliation path is scheduler storage sync, recorded as:
--
--       marie_scheduler.job_attempt.terminal_source = 'storage_sync'
--
-- Important:
--   This test intentionally edits the shared job-info KV row and extends the
--   scheduler run lease for the selected active job. By default mutation is
--   disabled. Set enable_mutation to TRUE only during a controlled HA test.
--
-- Current scheduler timing:
--   - _sync() polls every 60 seconds.
--   - _sync_terminal_job_state() ignores terminal job-info rows until their
--     end_time is older than 300 seconds.
--
-- To keep the test short, this script writes a synthetic terminal end_time
-- that is already 10 minutes old.

-- Setup: define the test parameters.
-- For the first run, leave target_job_id NULL and enable_mutation FALSE. The
-- script will show the latest active dispatched job that can be used.
--
-- To inject the mismatch, set enable_mutation TRUE and optionally paste a
-- specific target_job_id. If target_job_id stays NULL, the latest active
-- dispatched job is selected.
DROP TABLE IF EXISTS pg_temp.ha_lost_event_params;

CREATE TEMP TABLE ha_lost_event_params AS
SELECT
    NULL::uuid AS target_job_id,
    TRUE::boolean AS enable_mutation,
    INTERVAL '5 minutes' AS extend_run_lease_by,
    INTERVAL '10 minutes' AS synthetic_terminal_age;

-- Example mutation setup:
--
-- CREATE TEMP TABLE ha_lost_event_params AS
-- SELECT
--     '00000000-0000-0000-0000-000000000000'::uuid AS target_job_id,
--     TRUE::boolean AS enable_mutation,
--     INTERVAL '5 minutes' AS extend_run_lease_by,
--     INTERVAL '10 minutes' AS synthetic_terminal_age;

-- Query: show effective parameters. If enable_mutation is false, persistent
-- scheduler state will not be changed by this file.
SELECT
    target_job_id,
    enable_mutation,
    extend_run_lease_by,
    synthetic_terminal_age
FROM ha_lost_event_params;

-- Setup: choose the target job attempt.
--
-- With target_job_id set, this selects that job in any state so the same file
-- can be rerun for verification after reconciliation. With target_job_id NULL,
-- this selects the newest active dispatched job.
DROP TABLE IF EXISTS pg_temp.ha_lost_event_target;

CREATE TEMP TABLE ha_lost_event_target AS
SELECT
    j.id AS job_id,
    j.name AS job_name,
    j.dag_id,
    j.state::text AS job_state,
    j.run_owner,
    j.run_attempt_id,
    j.run_lease_expires_at,
    ja.gateway_instance_id AS activated_gateway_instance_id,
    ja.scheduler_lease_owner,
    ja.dispatch_started_at,
    ja.dispatch_confirmed_at,
    ja.terminal_source,
    ja.terminal_accepted,
    ja.terminal_gateway_instance_id,
    ja.terminal_at
FROM marie_scheduler.job j
JOIN marie_scheduler.job_attempt ja
  ON ja.job_id = j.id
 AND ja.run_attempt_id = j.run_attempt_id
CROSS JOIN ha_lost_event_params p
WHERE (
        p.target_job_id IS NOT NULL
        AND j.id = p.target_job_id
    )
   OR (
        p.target_job_id IS NULL
        AND j.state::text = 'active'
        AND ja.dispatch_confirmed_at IS NOT NULL
        AND j.run_owner IS NOT NULL
        AND j.run_attempt_id IS NOT NULL
    )
ORDER BY
    CASE WHEN p.target_job_id IS NOT NULL THEN 0 ELSE 1 END,
    ja.dispatch_confirmed_at DESC NULLS LAST,
    j.started_on DESC NULLS LAST
LIMIT 1;

-- Query: show the selected target. For mutation, this must be one active
-- dispatched job with run_owner and run_attempt_id populated.
SELECT *
FROM ha_lost_event_target;

-- Query: show the scheduler DB state beside durable job-info KV state before
-- any mutation. The lost-event test needs DB state active while KV status is
-- made terminal.
SELECT
    t.job_id,
    t.job_state,
    t.run_owner,
    t.run_attempt_id,
    t.run_lease_expires_at,
    kv.value->>'status' AS kv_status,
    to_timestamp(((kv.value->>'end_time')::bigint) / 1000.0) AS kv_end_time,
    kv.value->>'run_owner' AS kv_run_owner,
    kv.value->>'run_attempt_id' AS kv_run_attempt_id,
    kv.updated_at AS kv_updated_at
FROM ha_lost_event_target t
LEFT JOIN marie_scheduler.kv_store_worker kv
  ON kv.namespace = 'job'
 AND kv.key = 'marie_internal/job_info_' || t.job_id::text
 AND kv.is_deleted = FALSE;

-- Mutation: extend the scheduler run lease so expired-run recovery does not
-- race the storage-sync reconciliation path during this synthetic test.
--
-- This statement updates zero rows unless enable_mutation is TRUE.
UPDATE marie_scheduler.job j
SET run_lease_expires_at = now() + p.extend_run_lease_by
FROM ha_lost_event_target t
CROSS JOIN ha_lost_event_params p
WHERE p.enable_mutation IS TRUE
  AND j.id = t.job_id
  AND j.state::text = 'active'
RETURNING
    j.id AS job_id,
    j.state::text AS job_state,
    j.run_owner,
    j.run_attempt_id,
    j.run_lease_expires_at;

-- Mutation: simulate "durable terminal state was written, but the scheduler's
-- local terminal event was lost" by editing KV directly. This bypasses
-- JobInfoStorageClientProxy.put_status(), so no gateway event is published.
--
-- This statement updates zero rows unless enable_mutation is TRUE.
UPDATE marie_scheduler.kv_store_worker kv
SET
    value = COALESCE(kv.value, '{}'::jsonb) || jsonb_build_object(
        'status', 'SUCCEEDED',
        'message', 'synthetic lost local terminal event test',
        'end_time', (
            (extract(epoch FROM now() - p.synthetic_terminal_age) * 1000)::bigint
        ),
        'run_owner', t.run_owner,
        'run_attempt_id', t.run_attempt_id
    ),
    updated_at = now()
FROM ha_lost_event_target t
CROSS JOIN ha_lost_event_params p
WHERE p.enable_mutation IS TRUE
  AND kv.namespace = 'job'
  AND kv.key = 'marie_internal/job_info_' || t.job_id::text
  AND kv.is_deleted = FALSE
RETURNING
    t.job_id,
    kv.key,
    kv.value->>'status' AS kv_status,
    to_timestamp(((kv.value->>'end_time')::bigint) / 1000.0) AS kv_end_time,
    kv.value->>'run_owner' AS kv_run_owner,
    kv.value->>'run_attempt_id' AS kv_run_attempt_id,
    kv.updated_at AS kv_updated_at;

-- Query: immediate post-mutation state. After mutation, this should show
-- scheduler DB state still active while KV says SUCCEEDED.
SELECT
    j.id AS job_id,
    j.state::text AS job_state,
    j.completed_on,
    j.run_owner,
    j.run_attempt_id,
    j.run_lease_expires_at,
    kv.value->>'status' AS kv_status,
    to_timestamp(((kv.value->>'end_time')::bigint) / 1000.0) AS kv_end_time,
    kv.value->>'run_owner' AS kv_run_owner,
    kv.value->>'run_attempt_id' AS kv_run_attempt_id
FROM ha_lost_event_target t
JOIN marie_scheduler.job j ON j.id = t.job_id
LEFT JOIN marie_scheduler.kv_store_worker kv
  ON kv.namespace = 'job'
 AND kv.key = 'marie_internal/job_info_' || t.job_id::text
 AND kv.is_deleted = FALSE;

-- Manual step:
--   Wait for one or two scheduler sync cycles, usually 60-120 seconds.
--   Then rerun this file with enable_mutation FALSE and target_job_id set to
--   the same job_id, or rerun only the verification queries below in the same
--   database session.

-- Query: final reconciliation state. Expected result after sync:
--
--   job_state = completed
--   terminal_source = storage_sync
--   terminal_accepted = true
--   terminal_gateway_instance_id IS NOT NULL
--   terminal_reject_reason IS NULL
SELECT
    j.id AS job_id,
    j.state::text AS job_state,
    j.completed_on,
    ja.gateway_instance_id AS activated_gateway_instance_id,
    ja.scheduler_lease_owner,
    ja.terminal_gateway_instance_id,
    ja.terminal_source,
    ja.terminal_accepted,
    ja.terminal_reject_reason,
    ja.terminal_at,
    kv.value->>'status' AS kv_status,
    to_timestamp(((kv.value->>'end_time')::bigint) / 1000.0) AS kv_end_time
FROM ha_lost_event_target t
JOIN marie_scheduler.job j ON j.id = t.job_id
JOIN marie_scheduler.job_attempt ja
  ON ja.job_id = j.id
 AND ja.run_attempt_id = j.run_attempt_id
LEFT JOIN marie_scheduler.kv_store_worker kv
  ON kv.namespace = 'job'
 AND kv.key = 'marie_internal/job_info_' || t.job_id::text
 AND kv.is_deleted = FALSE;

-- Query: PASS/FAIL summary for the lost-local-event reconciliation path.
-- A PASS means the scheduler reconciled durable terminal storage state through
-- storage_sync rather than the normal job_event path.
WITH target_state AS (
    SELECT
        j.id AS job_id,
        j.state::text AS job_state,
        ja.terminal_source,
        ja.terminal_accepted,
        ja.terminal_reject_reason,
        ja.terminal_gateway_instance_id
    FROM ha_lost_event_target t
    JOIN marie_scheduler.job j ON j.id = t.job_id
    JOIN marie_scheduler.job_attempt ja
      ON ja.job_id = j.id
     AND ja.run_attempt_id = j.run_attempt_id
)
SELECT
    'lost_local_terminal_event_reconciled' AS check_name,
    CASE
        WHEN COUNT(*) = 1
         AND bool_and(job_state = 'completed')
         AND bool_and(terminal_source = 'storage_sync')
         AND bool_and(terminal_accepted IS TRUE)
         AND bool_and(terminal_reject_reason IS NULL)
         AND bool_and(terminal_gateway_instance_id IS NOT NULL)
        THEN 0
        ELSE 1
    END AS bad_rows,
    CASE
        WHEN COUNT(*) = 1
         AND bool_and(job_state = 'completed')
         AND bool_and(terminal_source = 'storage_sync')
         AND bool_and(terminal_accepted IS TRUE)
         AND bool_and(terminal_reject_reason IS NULL)
         AND bool_and(terminal_gateway_instance_id IS NOT NULL)
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result,
    'DB job must complete through terminal_source=storage_sync after a lost local terminal event.' AS expectation
FROM target_state;

-- Query: diagnostic for the case where the normal event path won instead of
-- the storage-sync path. If terminal_source is job_event, this test did not
-- prove lost-local-event reconciliation.
SELECT
    j.id AS job_id,
    j.state::text AS job_state,
    ja.terminal_source,
    ja.terminal_accepted,
    CASE
        WHEN ja.terminal_source = 'job_event'
            THEN 'normal job_event path won; rerun with direct KV mutation before event handling'
        WHEN ja.terminal_source = 'storage_sync'
            THEN 'storage sync path proved'
        WHEN ja.terminal_source IS NULL
            THEN 'not reconciled yet; wait another sync cycle or inspect scheduler logs'
        ELSE 'unexpected terminal_source; inspect terminal audit details'
    END AS interpretation
FROM ha_lost_event_target t
JOIN marie_scheduler.job j ON j.id = t.job_id
JOIN marie_scheduler.job_attempt ja
  ON ja.job_id = j.id
 AND ja.run_attempt_id = j.run_attempt_id;
