-- File: 006_job_history.sql
-- Description: Job state change history table
-- Dependencies: 001_schema.sql, 002_enums.sql (job_state enum)

CREATE TABLE IF NOT EXISTS {schema}.job_history (
    history_id BIGSERIAL PRIMARY KEY,
    id UUID NOT NULL,
    name TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    data JSONB,
    state {schema}.job_state NOT NULL,
    retry_limit INTEGER NOT NULL DEFAULT 2,
    retry_count INTEGER NOT NULL DEFAULT 0,
    retry_delay INTEGER NOT NULL DEFAULT 0,
    retry_backoff BOOLEAN NOT NULL DEFAULT FALSE,
    start_after TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expire_in INTERVAL NOT NULL DEFAULT INTERVAL '15 minutes',
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_on TIMESTAMP WITH TIME ZONE,
    completed_on TIMESTAMP WITH TIME ZONE,
    keep_until TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW() + INTERVAL '14 days',
    output JSONB,
    dead_letter TEXT,
    policy TEXT,
    duration INTERVAL,
    sla_interval INTERVAL,
    soft_sla TIMESTAMP WITH TIME ZONE,
    hard_sla TIMESTAMP WITH TIME ZONE,
    sla_miss_logged BOOLEAN NOT NULL DEFAULT FALSE,
    dag_id UUID NOT NULL,
    job_level INTEGER NOT NULL DEFAULT 0,
    dependencies JSONB DEFAULT '[]'::JSONB,
    branch_metadata JSONB,
    history_created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE {schema}.job_history IS 'Audit trail for job state changes';
