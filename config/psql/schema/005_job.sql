-- File: 005_job.sql
-- Description: Unpartitioned active scheduler job table
-- Dependencies: 001_schema.sql, 002_enums.sql (job_state enum)

CREATE TABLE IF NOT EXISTS {schema}.job (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    data JSONB,
    state {schema}.job_state NOT NULL DEFAULT 'created',
    retry_limit INTEGER NOT NULL DEFAULT 2,
    retry_count INTEGER NOT NULL DEFAULT 0,
    retry_delay INTEGER NOT NULL DEFAULT 0,
    retry_backoff BOOLEAN NOT NULL DEFAULT FALSE,
    start_after TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_on TIMESTAMP WITH TIME ZONE,
    expire_in INTERVAL NOT NULL DEFAULT INTERVAL '15 minutes',
    created_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_on TIMESTAMP WITH TIME ZONE,
    keep_until TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW() + INTERVAL '14 days',
    output JSONB,
    dead_letter TEXT,
    policy TEXT,
    dependencies JSONB DEFAULT '[]'::JSONB,
    dag_id UUID NOT NULL,
    job_level INTEGER NOT NULL DEFAULT 0,
    duration INTERVAL,
    sla_interval INTERVAL,
    soft_sla TIMESTAMP WITH TIME ZONE,
    hard_sla TIMESTAMP WITH TIME ZONE,
    sla_miss_logged BOOLEAN NOT NULL DEFAULT FALSE,
    branch_metadata JSONB,
    -- Lease columns for job acquisition
    lease_owner TEXT,
    lease_expires_at TIMESTAMP WITH TIME ZONE,
    lease_epoch BIGINT DEFAULT 0,
    run_owner TEXT,
    run_attempt_id UUID,
    run_lease_expires_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (id),
    UNIQUE (name, id)
);

COMMENT ON TABLE {schema}.job IS
    'Unpartitioned active scheduler jobs; queue names remain logical routing metadata';
