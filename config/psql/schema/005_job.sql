-- File: 005_job.sql
-- Description: Main job queue table with list partitioning
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
    -- Lease columns for job acquisition
    lease_owner TEXT,
    lease_expires_at TIMESTAMP WITH TIME ZONE,
    lease_epoch BIGINT DEFAULT 0,
    run_owner TEXT,
    run_lease_expires_at TIMESTAMP WITH TIME ZONE
) PARTITION BY LIST (name);

-- Primary key added separately to support partitioning (idempotent)
DO $$ BEGIN
    ALTER TABLE {schema}.job ADD PRIMARY KEY (name, id);
EXCEPTION
    WHEN duplicate_object THEN NULL;
    WHEN duplicate_table THEN NULL;
END $$;

COMMENT ON TABLE {schema}.job IS 'Main job queue table - partitioned by queue name';
