-- File: 012_job_history_trigger.sql
-- Description: Trigger function and trigger for job history tracking
-- Dependencies: 005_job.sql, 006_job_history.sql
--
-- History records are created for:
--   - Every INSERT (new job creation)
--   - UPDATEs that change meaningful state columns (state, retry_count,
--     output, completed_on, started_on, branch_metadata)
--
-- Duration-only updates (from pg_cron refresh_job_durations) and
-- lease-only updates (lease_owner, lease_expires_at, run_owner, etc.)
-- are intentionally excluded to prevent history table bloat.

-- Create the trigger function that populates job_history (idempotent)
CREATE OR REPLACE FUNCTION {schema}.job_update_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO {schema}.job_history (
        id, name, priority, data, state, retry_limit, retry_count, retry_delay,
        retry_backoff, start_after, expire_in, created_on, started_on,
        completed_on, keep_until, output, dead_letter, policy, duration,
        sla_interval, soft_sla, hard_sla, sla_miss_logged,
        dag_id, job_level, dependencies, branch_metadata, history_created_on
    )
    VALUES (
        NEW.id, NEW.name, NEW.priority, NEW.data, NEW.state, NEW.retry_limit,
        NEW.retry_count, NEW.retry_delay, NEW.retry_backoff, NEW.start_after,
        NEW.expire_in, NEW.created_on, NEW.started_on, NEW.completed_on,
        NEW.keep_until, NEW.output, NEW.dead_letter, NEW.policy, NEW.duration,
        NEW.sla_interval, NEW.soft_sla, NEW.hard_sla, NEW.sla_miss_logged,
        NEW.dag_id, NEW.job_level, NEW.dependencies, NEW.branch_metadata, NOW()
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop the old unconditional trigger
DROP TRIGGER IF EXISTS job_update_trigger ON {schema}.job;

-- INSERT trigger: always fire for new job creation
DROP TRIGGER IF EXISTS job_insert_trigger ON {schema}.job;
CREATE TRIGGER job_insert_trigger
AFTER INSERT ON {schema}.job
FOR EACH ROW
EXECUTE FUNCTION {schema}.job_update_trigger_function();

-- UPDATE trigger: only fire on meaningful state changes
-- Skips duration-only updates (refresh_job_durations cron) and
-- lease-only updates (lease/release/extend operations)
DROP TRIGGER IF EXISTS job_update_state_trigger ON {schema}.job;
CREATE TRIGGER job_update_state_trigger
AFTER UPDATE ON {schema}.job
FOR EACH ROW
WHEN (
    OLD.state IS DISTINCT FROM NEW.state
    OR OLD.retry_count IS DISTINCT FROM NEW.retry_count
    OR OLD.output IS DISTINCT FROM NEW.output
    OR OLD.completed_on IS DISTINCT FROM NEW.completed_on
    OR OLD.started_on IS DISTINCT FROM NEW.started_on
    OR OLD.branch_metadata IS DISTINCT FROM NEW.branch_metadata
)
EXECUTE FUNCTION {schema}.job_update_trigger_function();
