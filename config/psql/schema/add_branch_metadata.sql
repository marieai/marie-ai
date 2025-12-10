-- Migration: Add branch_metadata column for branch execution tracking
-- Date: 2025-11-13
-- Description: Adds a dedicated JSONB column to track branch/switch node execution metadata

-- Add branch_metadata column to job table
ALTER TABLE marie_scheduler.job ADD COLUMN IF NOT EXISTS branch_metadata JSONB;

-- Add branch_metadata column to job_history table
ALTER TABLE marie_scheduler.job_history ADD COLUMN IF NOT EXISTS branch_metadata JSONB;

-- Add comment explaining the column
COMMENT ON COLUMN marie_scheduler.job.branch_metadata IS
'Branch execution metadata for tracking conditional execution flow.
Contains node_type, selected_path_ids, skip_reason, etc.';

COMMENT ON COLUMN marie_scheduler.job_history.branch_metadata IS
'Historical branch execution metadata for audit and debugging.';

-- Create GIN index for efficient JSON queries
CREATE INDEX IF NOT EXISTS job_branch_metadata_idx
ON marie_scheduler.job USING gin(branch_metadata);

-- Create partial index for quickly finding skipped jobs
CREATE INDEX IF NOT EXISTS job_branch_skipped_idx
ON marie_scheduler.job ((branch_metadata->>'skipped'))
WHERE branch_metadata->>'skipped' = 'true';

-- Create partial index for finding BRANCH/SWITCH nodes
CREATE INDEX IF NOT EXISTS job_branch_node_type_idx
ON marie_scheduler.job ((branch_metadata->>'node_type'))
WHERE branch_metadata->>'node_type' IN ('BRANCH', 'SWITCH');

-- Update the trigger function to include branch_metadata
CREATE OR REPLACE FUNCTION marie_scheduler.job_update_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO marie_scheduler.job_history (
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
        NEW.dag_id, NEW.job_level, NEW.dependencies, NEW.branch_metadata, now()
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Verify migration
DO $$
DECLARE
    column_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'marie_scheduler'
        AND table_name = 'job'
        AND column_name = 'branch_metadata'
    ) INTO column_exists;

    IF column_exists THEN
        RAISE NOTICE 'Migration successful: branch_metadata column added to marie_scheduler.job';
    ELSE
        RAISE EXCEPTION 'Migration failed: branch_metadata column not found';
    END IF;
END $$;
