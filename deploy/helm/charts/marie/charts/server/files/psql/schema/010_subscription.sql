-- File: 010_subscription.sql
-- Description: Event subscription table for pub/sub messaging
-- Dependencies: 001_schema.sql

CREATE TABLE IF NOT EXISTS {schema}.subscription (
    event TEXT NOT NULL,
    name TEXT NOT NULL,
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (event, name)
);

COMMENT ON TABLE {schema}.subscription IS 'Event subscriptions for pub/sub messaging';
