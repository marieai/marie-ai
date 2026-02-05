-- Sensor/Trigger System Enums
-- Schema: marie_scheduler (backend-owned)
-- Related: sensor-trigger-system-design.md

-- Sensor types enum (matches n8n trigger patterns)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'sensor_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = '{schema}')) THEN
        CREATE TYPE {schema}.sensor_type AS ENUM (
            'manual',      -- Click-to-run for testing
            'schedule',    -- Cron/time-based
            'webhook',     -- HTTP endpoint (ingest to event_log)
            'polling',     -- External API polling
            'event',       -- Message queue (RabbitMQ/Kafka, ingest to event_log)
            'run_status',  -- Job completion monitoring
            'asset'        -- Asset materialization
        );
    END IF;
END $$;

-- Sensor operational status enum
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'sensor_status' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = '{schema}')) THEN
        CREATE TYPE {schema}.sensor_status AS ENUM (
            'active',      -- Running, evaluating
            'inactive',    -- Stopped, not evaluating
            'paused',      -- Temporarily paused
            'error'        -- Failed, needs attention
        );
    END IF;
END $$;

-- Tick status enum (evaluation result)
-- Includes 'started' for two-phase commit / crash recovery
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tick_status' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = '{schema}')) THEN
        CREATE TYPE {schema}.tick_status AS ENUM (
            'started',     -- In progress (for crash recovery)
            'success',     -- Evaluated successfully, may have fired jobs
            'skipped',     -- Evaluated, no action needed (e.g., cron not due)
            'failed'       -- Evaluation error
        );
    END IF;
END $$;
