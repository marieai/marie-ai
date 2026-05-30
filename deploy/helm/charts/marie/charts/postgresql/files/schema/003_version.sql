-- File: 003_version.sql
-- Description: Schema version tracking table
-- Dependencies: 001_schema.sql

CREATE TABLE IF NOT EXISTS {schema}.version (
    version INT PRIMARY KEY,
    maintained_on TIMESTAMP WITH TIME ZONE,
    cron_on TIMESTAMP WITH TIME ZONE,
    monitored_on TIMESTAMP WITH TIME ZONE
);

COMMENT ON TABLE {schema}.version IS 'Tracks schema version and maintenance timestamps';
