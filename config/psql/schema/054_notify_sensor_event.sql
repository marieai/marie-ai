-- Sensor Event Notification Trigger
-- Schema: marie_scheduler (backend-owned)
-- Related: sensor-trigger-system-design.md
--
-- This trigger sends a PostgreSQL NOTIFY on event_log INSERT.
-- The SensorWorker listens for these notifications for fast-path wake-up.
--
-- IMPORTANT: NOTIFY is best-effort optimization only.
-- Polling remains authoritative for correctness and recovery.
-- Payload is minimal (IDs only) to respect 8KB limit.

CREATE OR REPLACE FUNCTION {schema}.notify_sensor_event() RETURNS trigger AS $$
DECLARE
    payload TEXT;
BEGIN
    -- Minimal payload: only IDs (respects 8KB NOTIFY limit)
    payload := json_build_object(
        'event_log_id', NEW.event_log_id,
        'sensor_external_id', NEW.sensor_external_id,
        'source', NEW.source
    )::text;

    -- Send notification to 'sensor_event' channel
    PERFORM pg_notify('sensor_event', payload);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on event_log INSERT
DROP TRIGGER IF EXISTS trg_event_log_notify ON {schema}.event_log;

CREATE TRIGGER trg_event_log_notify
AFTER INSERT ON {schema}.event_log
FOR EACH ROW
EXECUTE FUNCTION {schema}.notify_sensor_event();

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON FUNCTION {schema}.notify_sensor_event() IS
    'Sends PostgreSQL NOTIFY on event_log INSERT for fast-path sensor wake-up. '
    'Payload contains only IDs to stay within 8KB limit. '
    'Delivery is best-effort; polling catches missed notifications.';
