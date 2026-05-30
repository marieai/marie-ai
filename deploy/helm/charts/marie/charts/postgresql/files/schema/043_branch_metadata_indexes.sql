-- File: 043_branch_metadata_indexes.sql
-- Migration: Add branch_metadata indexes and comments
-- Date: 2025-11-13
-- Description: Adds indexes and documentation for branch_metadata column
-- Dependencies: 005_job.sql, 006_job_history.sql

-- Add comment explaining the column
COMMENT ON COLUMN {schema}.job.branch_metadata IS
'Branch execution metadata for tracking conditional execution flow.
Contains node_type, selected_path_ids, skip_reason, etc.';

COMMENT ON COLUMN {schema}.job_history.branch_metadata IS
'Historical branch execution metadata for audit and debugging.';

-- Create GIN index for efficient JSON queries
CREATE INDEX IF NOT EXISTS job_branch_metadata_idx
ON {schema}.job USING gin(branch_metadata);

-- Create partial index for quickly finding skipped jobs
CREATE INDEX IF NOT EXISTS job_branch_skipped_idx
ON {schema}.job ((branch_metadata->>'skipped'))
WHERE branch_metadata->>'skipped' = 'true';

-- Create partial index for finding BRANCH/SWITCH nodes
CREATE INDEX IF NOT EXISTS job_branch_node_type_idx
ON {schema}.job ((branch_metadata->>'node_type'))
WHERE branch_metadata->>'node_type' IN ('BRANCH', 'SWITCH');
