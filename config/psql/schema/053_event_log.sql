-- Durable Event Log Table
-- Schema: marie_scheduler (backend-owned)
-- Related: sensor-trigger-system-design.md
--
-- All push events (webhooks, message queues) become data in this log.
-- The sensor daemon polls this log for event/webhook sensors.
-- This provides a single durability surface for all event sources.

CREATE TABLE IF NOT EXISTS {schema}.event_log (
    -- Monotonic cursor for reliable polling
    event_log_id BIGSERIAL PRIMARY KEY,

    -- Event identity
    event_id UUID NOT NULL DEFAULT gen_random_uuid(),

    -- Source metadata
    source TEXT NOT NULL,                           -- 'webhook', 'rabbitmq', 'kafka', 'internal'
    sensor_type {schema}.sensor_type,               -- Optional routing hint
    sensor_external_id UUID,                        -- Optional target trigger_config id

    -- Routing
    routing_key TEXT,                               -- Queue routing key or webhook path
    event_key TEXT,                                 -- Stable event key from source (for dedup)

    -- Payload
    payload JSONB NOT NULL,
    headers JSONB NOT NULL DEFAULT '{}',

    -- Timing
    received_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Primary cursor-based polling (most important)
CREATE INDEX IF NOT EXISTS idx_event_log_id ON {schema}.event_log(event_log_id);

-- Filter by source
CREATE INDEX IF NOT EXISTS idx_event_log_source ON {schema}.event_log(source);

-- Filter by target sensor (for sensor-specific event queries)
CREATE INDEX IF NOT EXISTS idx_event_log_sensor_ext
    ON {schema}.event_log(sensor_external_id)
    WHERE sensor_external_id IS NOT NULL;

-- Combined cursor + sensor for efficient polling per sensor
CREATE INDEX IF NOT EXISTS idx_event_log_sensor_cursor
    ON {schema}.event_log(sensor_external_id, event_log_id)
    WHERE sensor_external_id IS NOT NULL;

-- Event key deduplication lookup
CREATE INDEX IF NOT EXISTS idx_event_log_event_key
    ON {schema}.event_log(event_key)
    WHERE event_key IS NOT NULL;

-- Retention cleanup (received_at for time-based pruning)
CREATE INDEX IF NOT EXISTS idx_event_log_retention
    ON {schema}.event_log(received_at);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE {schema}.event_log IS
    'Durable event log for webhook and message queue events. '
    'All push events are written here before acknowledgment. '
    'Sensor daemon polls this table using cursor-based pagination.';

COMMENT ON COLUMN {schema}.event_log.event_log_id IS
    'Monotonically increasing ID used as cursor for reliable polling.';

COMMENT ON COLUMN {schema}.event_log.event_key IS
    'Stable event key from source (e.g., message_id, delivery_id) for deduplication. '
    'If not available, computed as hash of payload + routing_key + time bucket.';

COMMENT ON COLUMN {schema}.event_log.sensor_external_id IS
    'Optional link to specific trigger_config. '
    'NULL means event will be processed by all matching sensors.';
