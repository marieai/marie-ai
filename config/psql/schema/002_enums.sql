-- File: 002_enums.sql
-- Description: All enum types for the scheduler schema
-- Dependencies: 001_schema.sql

-- Create job_state enum if it doesn't exist
DO $$ BEGIN
    CREATE TYPE {schema}.job_state AS ENUM (
        'created',
        'retry',
        'active',
        'completed',
        'expired',
        'cancelled',
        'failed'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
