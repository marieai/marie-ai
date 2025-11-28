-- File: 009_schedule.sql
-- Description: Cron-based job scheduling table
-- Dependencies: 001_schema.sql

CREATE TABLE IF NOT EXISTS {schema}.schedule (
    name TEXT PRIMARY KEY,
    cron TEXT NOT NULL,
    timezone TEXT,
    data JSONB,
    options JSONB,
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE {schema}.schedule IS 'Cron-based job schedule definitions';
