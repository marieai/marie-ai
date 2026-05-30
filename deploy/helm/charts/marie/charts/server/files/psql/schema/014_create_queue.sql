-- File: 014_create_queue.sql
-- Description: Function to create a new job queue with partition
-- Dependencies: 004_queue.sql, 005_job.sql

-- Create or replace the queue creation function (idempotent)
CREATE OR REPLACE FUNCTION {schema}.create_queue(queue_name TEXT, options JSON)
RETURNS VOID AS
$$
DECLARE
    table_name VARCHAR := 'j' || encode(sha224(queue_name::bytea), 'hex');
    queue_created_on TIMESTAMPTZ;
BEGIN
    -- Insert queue configuration
    WITH q AS (
        INSERT INTO {schema}.queue (
            name,
            policy,
            retry_limit,
            retry_delay,
            retry_backoff,
            expire_seconds,
            retention_minutes,
            dead_letter,
            partition_name
        )
        VALUES (
            queue_name,
            options->>'policy',
            (options->>'retry_limit')::INT,
            (options->>'retry_delay')::INT,
            (options->>'retry_backoff')::BOOL,
            (options->>'expire_in_seconds')::INT,
            (options->>'retention_minutes')::INT,
            options->>'dead_letter',
            table_name
        )
        ON CONFLICT DO NOTHING
        RETURNING created_on
    )
    SELECT created_on INTO queue_created_on FROM q;

    -- If queue already exists, return early
    IF queue_created_on IS NULL THEN
        RETURN;
    END IF;

    -- Create partition table for this queue (INCLUDING GENERATED for generated columns)
    EXECUTE format('CREATE TABLE {schema}.%I (LIKE {schema}.job INCLUDING DEFAULTS INCLUDING GENERATED)', table_name);
    EXECUTE format('ALTER TABLE {schema}.%1$I ADD PRIMARY KEY (name, id)', table_name);
    EXECUTE format('ALTER TABLE {schema}.%I ADD CONSTRAINT cjc CHECK (name=%L)', table_name, queue_name);
    EXECUTE format('ALTER TABLE {schema}.job ATTACH PARTITION {schema}.%I FOR VALUES IN (%L)', table_name, queue_name);
END;
$$
LANGUAGE plpgsql;
