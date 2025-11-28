-- File: 043_add_branch_metadata.sql
-- Description: Add branch_metadata column for workflow branching support
-- Dependencies: 005_job.sql

-- Add branch_metadata column if it doesn't exist
DO $$ BEGIN
    ALTER TABLE {schema}.job ADD COLUMN branch_metadata JSONB;
EXCEPTION
    WHEN duplicate_column THEN NULL;
END $$;

COMMENT ON COLUMN {schema}.job.branch_metadata IS 'Stores branch/switch execution metadata for workflow branching';
