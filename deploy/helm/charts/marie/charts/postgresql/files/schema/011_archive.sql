-- File: 011_archive.sql
-- Description: Archived jobs table (cloned from job table structure)
-- Dependencies: 001_schema.sql, 005_job.sql

-- Create archive table with same structure as job table (idempotent)
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = '{schema}' AND table_name = 'archive') THEN
        CREATE TABLE {schema}.archive (LIKE {schema}.job INCLUDING ALL);
    END IF;
END $$;

-- Add archived_on timestamp column if it doesn't exist
DO $$ BEGIN
    ALTER TABLE {schema}.archive ADD COLUMN archived_on TIMESTAMPTZ NOT NULL DEFAULT NOW();
EXCEPTION
    WHEN duplicate_column THEN NULL;
END $$;

-- Create indexes for efficient archive queries (idempotent)
CREATE INDEX IF NOT EXISTS archive_archivedon_idx ON {schema}.archive(archived_on);
CREATE INDEX IF NOT EXISTS archive_id_idx ON {schema}.archive(id);

COMMENT ON TABLE {schema}.archive IS 'Archived completed/expired jobs for historical retention';
