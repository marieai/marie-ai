-- File: 065_job_attempt.sql
-- Description: Durable per-attempt audit trail for gateway HA debugging
-- Dependencies: 001_schema.sql, 005_job.sql

ALTER TABLE {schema}.job
  ADD COLUMN IF NOT EXISTS run_attempt_id UUID;

CREATE TABLE IF NOT EXISTS {schema}.job_attempt (
    run_attempt_id UUID PRIMARY KEY,
    job_id UUID NOT NULL,
    job_name TEXT NOT NULL,
    dag_id UUID NOT NULL,
    run_owner TEXT NOT NULL,
    scheduler_lease_owner TEXT NOT NULL,
    gateway_instance_id TEXT,
    executor TEXT,
    attempt_state TEXT NOT NULL DEFAULT 'activated',
    activated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    dispatch_started_at TIMESTAMP WITH TIME ZONE,
    dispatch_confirmed_at TIMESTAMP WITH TIME ZONE,
    dispatch_error TEXT,
    terminal_at TIMESTAMP WITH TIME ZONE,
    terminal_status TEXT,
    terminal_work_state TEXT,
    terminal_source TEXT,
    terminal_gateway_instance_id TEXT,
    terminal_scheduler_lease_owner TEXT,
    terminal_accepted BOOLEAN,
    terminal_reject_reason TEXT,
    recovery_at TIMESTAMP WITH TIME ZONE,
    recovery_state TEXT,
    recovery_reason TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS job_attempt_job_id_idx
    ON {schema}.job_attempt (job_id);

CREATE INDEX IF NOT EXISTS job_attempt_dag_id_idx
    ON {schema}.job_attempt (dag_id);

CREATE INDEX IF NOT EXISTS job_attempt_run_owner_idx
    ON {schema}.job_attempt (run_owner);

CREATE INDEX IF NOT EXISTS job_attempt_gateway_instance_idx
    ON {schema}.job_attempt (gateway_instance_id);

CREATE INDEX IF NOT EXISTS job_attempt_terminal_gateway_idx
    ON {schema}.job_attempt (terminal_gateway_instance_id);

COMMENT ON TABLE {schema}.job_attempt IS
    'Durable audit record for each scheduler run attempt, used to answer which gateway activated, recovered, or terminally handled a job.';

COMMENT ON COLUMN {schema}.job_attempt.dispatch_started_at IS
    'Legacy rolling-upgrade field; current gateways use scheduler trace events for dispatch diagnostics.';
COMMENT ON COLUMN {schema}.job_attempt.dispatch_confirmed_at IS
    'Legacy rolling-upgrade field; current gateways use scheduler trace events for dispatch diagnostics.';
COMMENT ON COLUMN {schema}.job_attempt.dispatch_error IS
    'Legacy rolling-upgrade field; current gateways use scheduler trace events for dispatch diagnostics.';

-- Backfill currently visible attempts so rolling upgrades have an audit row for
-- active or recently terminal work that already has a run_attempt_id.
INSERT INTO {schema}.job_attempt (
    run_attempt_id,
    job_id,
    job_name,
    dag_id,
    run_owner,
    scheduler_lease_owner,
    attempt_state,
    activated_at,
    terminal_at,
    terminal_status,
    terminal_work_state,
    terminal_accepted,
    metadata
)
SELECT
    j.run_attempt_id,
    j.id,
    j.name,
    j.dag_id,
    COALESCE(j.run_owner, 'unknown'),
    COALESCE(j.run_owner, 'unknown'),
    CASE
        WHEN j.state::TEXT = 'active' THEN 'activated'
        WHEN j.state::TEXT IN ('completed', 'failed', 'cancelled', 'expired', 'skipped') THEN j.state::TEXT
        ELSE 'observed'
    END,
    COALESCE(j.started_on, j.created_on, NOW()),
    CASE
        WHEN j.state::TEXT IN ('completed', 'failed', 'cancelled', 'expired', 'skipped') THEN j.completed_on
        ELSE NULL
    END,
    CASE
        WHEN j.state::TEXT IN ('completed', 'failed', 'cancelled', 'expired', 'skipped') THEN j.state::TEXT
        ELSE NULL
    END,
    CASE
        WHEN j.state::TEXT IN ('completed', 'failed', 'cancelled', 'expired', 'skipped') THEN j.state::TEXT
        ELSE NULL
    END,
    CASE
        WHEN j.state::TEXT IN ('completed', 'failed', 'cancelled', 'expired', 'skipped') THEN TRUE
        ELSE NULL
    END,
    jsonb_build_object('backfilled', TRUE)
FROM {schema}.job j
WHERE j.run_attempt_id IS NOT NULL
ON CONFLICT (run_attempt_id) DO NOTHING;

-- Repair attempts that were activated before terminal audit wiring was present
-- or while terminal audit writes were failing. This keeps the table useful after
-- a rolling upgrade, but labels the source as a job-state backfill because the
-- exact terminal gateway may no longer be knowable.
UPDATE {schema}.job_attempt ja
SET attempt_state = CASE
        WHEN ja.recovery_state IS NOT NULL THEN ja.attempt_state
        ELSE j.state::TEXT
    END,
    terminal_at = COALESCE(ja.terminal_at, j.completed_on, NOW()),
    terminal_status = COALESCE(ja.terminal_status, j.state::TEXT),
    terminal_work_state = COALESCE(ja.terminal_work_state, j.state::TEXT),
    terminal_source = COALESCE(ja.terminal_source, 'job_state_backfill'),
    terminal_gateway_instance_id = COALESCE(
        ja.terminal_gateway_instance_id,
        ja.gateway_instance_id
    ),
    terminal_scheduler_lease_owner = COALESCE(
        ja.terminal_scheduler_lease_owner,
        ja.scheduler_lease_owner
    ),
    terminal_accepted = COALESCE(ja.terminal_accepted, TRUE),
    updated_on = NOW()
FROM {schema}.job j
WHERE ja.job_id = j.id
  AND ja.run_attempt_id = j.run_attempt_id
  AND j.state::TEXT IN ('completed', 'failed', 'cancelled', 'expired', 'skipped')
  AND ja.terminal_at IS NULL;
