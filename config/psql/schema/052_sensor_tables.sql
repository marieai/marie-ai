-- Sensor/Trigger System Tables
-- Schema: marie_scheduler (backend-owned)
-- Related: sensor-trigger-system-design.md

-- Main sensor table (runtime state)
-- Synced from marie_studio.trigger_config via REST API
CREATE TABLE IF NOT EXISTS {schema}.sensor (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity (synced from marie-studio trigger_config)
    external_id UUID NOT NULL UNIQUE,  -- References marie_studio.trigger_config.id
    name TEXT NOT NULL,
    sensor_type {schema}.sensor_type NOT NULL,

    -- Configuration (JSON, copied from trigger_config)
    config JSONB NOT NULL DEFAULT '{}',

    -- Target (what to run when sensor fires)
    target_job_name TEXT,
    target_dag_id UUID,

    -- Status
    status {schema}.sensor_status NOT NULL DEFAULT 'inactive',

    -- Runtime state (managed by SensorWorker)
    cursor TEXT,                                    -- User-managed state (e.g., event_log_id)
    last_tick_at TIMESTAMP WITH TIME ZONE,          -- When last evaluated
    last_run_key TEXT,                              -- Most recent run key
    failure_count INTEGER NOT NULL DEFAULT 0,       -- Consecutive failures (reset on success)
    last_error TEXT,                                -- Last error message

    -- Timing
    minimum_interval_seconds INTEGER NOT NULL DEFAULT 30,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Sensor evaluation history (ticks)
-- Records every evaluation attempt (success, skipped, or failed)
CREATE TABLE IF NOT EXISTS {schema}.sensor_tick (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sensor_id UUID NOT NULL REFERENCES {schema}.sensor(id) ON DELETE CASCADE,

    status {schema}.tick_status NOT NULL,
    cursor TEXT,                          -- Cursor after evaluation

    -- Run reservation (for crash recovery - Dagster pattern)
    run_requests JSONB,                   -- Serialized RunRequest list
    reserved_run_ids UUID[] DEFAULT '{}', -- Pre-reserved job IDs

    -- Results
    run_ids UUID[] DEFAULT '{}',          -- Jobs actually submitted
    skip_reason TEXT,
    error_message TEXT,

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,

    -- Debug info
    trigger_payload JSONB
);

-- Run key tracking for idempotency
-- Prevents duplicate job submissions from the same sensor
CREATE TABLE IF NOT EXISTS {schema}.sensor_run_key (
    sensor_id UUID NOT NULL REFERENCES {schema}.sensor(id) ON DELETE CASCADE,
    run_key TEXT NOT NULL,
    job_id UUID,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    PRIMARY KEY (sensor_id, run_key)
);

-- Webhook registrations (path -> sensor mapping for ingestion)
-- Maps webhook paths to sensors for event_log routing
CREATE TABLE IF NOT EXISTS {schema}.webhook_registration (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sensor_id UUID NOT NULL REFERENCES {schema}.sensor(id) ON DELETE CASCADE,

    path TEXT NOT NULL UNIQUE,          -- e.g., /webhooks/github-push
    methods TEXT[] NOT NULL DEFAULT '{POST}',
    auth_type TEXT,                     -- 'none', 'api_key', 'hmac', 'basic'
    auth_secret TEXT,                   -- Encrypted secret for HMAC/basic

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Sensor lookups
CREATE INDEX IF NOT EXISTS idx_sensor_status ON {schema}.sensor(status);
CREATE INDEX IF NOT EXISTS idx_sensor_type ON {schema}.sensor(sensor_type);
CREATE INDEX IF NOT EXISTS idx_sensor_external ON {schema}.sensor(external_id);

-- Active sensors for daemon polling (status + last_tick for ordering)
CREATE INDEX IF NOT EXISTS idx_sensor_active_poll
    ON {schema}.sensor(status, last_tick_at ASC NULLS FIRST)
    WHERE status = 'active';

-- Tick lookups
CREATE INDEX IF NOT EXISTS idx_sensor_tick_sensor ON {schema}.sensor_tick(sensor_id);
CREATE INDEX IF NOT EXISTS idx_sensor_tick_started ON {schema}.sensor_tick(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_tick_sensor_started
    ON {schema}.sensor_tick(sensor_id, started_at DESC);

-- Stuck tick detection (STARTED ticks for crash recovery)
CREATE INDEX IF NOT EXISTS idx_sensor_tick_started_status
    ON {schema}.sensor_tick(status, started_at)
    WHERE status = 'started';

-- Run key lookups
CREATE INDEX IF NOT EXISTS idx_sensor_run_key_created ON {schema}.sensor_run_key(created_at);

-- CAUSES
--- ERROR: functions in index predicate must be marked IMMUTABLE


-- Retention cleanup indexes (for efficient pruning)
-- CREATE INDEX IF NOT EXISTS idx_sensor_tick_retention
--     ON {schema}.sensor_tick(started_at)
--     WHERE started_at < NOW() - INTERVAL '30 days';

-- CREATE INDEX IF NOT EXISTS idx_sensor_run_key_retention
--     ON {schema}.sensor_run_key(created_at)
--     WHERE created_at < NOW() - INTERVAL '30 days';
