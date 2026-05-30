-- File: 004_queue.sql
-- Description: Queue configuration table
-- Dependencies: 001_schema.sql

CREATE TABLE IF NOT EXISTS {schema}.queue (
    name TEXT PRIMARY KEY,
    policy TEXT,
    retry_limit INT,
    retry_delay INT,
    retry_backoff BOOL,
    expire_seconds INT,
    retention_minutes INT,
    dead_letter TEXT REFERENCES {schema}.queue (name),
    partition_name TEXT,
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE {schema}.queue IS 'Queue configuration and metadata for job partitions';
