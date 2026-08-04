-- File: 014_create_queue.sql
-- Description: Function to create logical queue metadata
-- Dependencies: 004_queue.sql, 005_job.sql

-- Create or replace the queue creation function (idempotent)
CREATE OR REPLACE FUNCTION {schema}.create_queue(queue_name TEXT, options JSON)
RETURNS VOID AS
$$
BEGIN
    INSERT INTO {schema}.queue (
        name,
        policy,
        retry_limit,
        retry_delay,
        retry_backoff,
        expire_seconds,
        retention_minutes,
        dead_letter
    )
    VALUES (
        queue_name,
        options->>'policy',
        (options->>'retry_limit')::INT,
        (options->>'retry_delay')::INT,
        (options->>'retry_backoff')::BOOL,
        (options->>'expire_in_seconds')::INT,
        (options->>'retention_minutes')::INT,
        options->>'dead_letter'
    )
    ON CONFLICT (name) DO NOTHING;
END;
$$
LANGUAGE plpgsql;
