-- Sensor Retention Cleanup Functions
-- Schema: marie_scheduler (backend-owned)
-- Related: sensor-trigger-system-design.md
--
-- These functions clean up old sensor data for retention management.
-- Called by MaintenanceService on a schedule.

-- ============================================================================
-- TICK CLEANUP
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.cleanup_old_ticks(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM {schema}.sensor_tick
    WHERE started_at < NOW() - (retention_days || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION {schema}.cleanup_old_ticks(INTEGER) IS
    'Deletes sensor_tick rows older than retention_days. '
    'Default retention: 30 days. Returns count of deleted rows.';

-- ============================================================================
-- RUN KEY CLEANUP
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.cleanup_old_run_keys(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM {schema}.sensor_run_key
    WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION {schema}.cleanup_old_run_keys(INTEGER) IS
    'Deletes sensor_run_key rows older than retention_days. '
    'Default retention: 30 days. Returns count of deleted rows.';

-- ============================================================================
-- EVENT LOG CLEANUP
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.cleanup_old_events(retention_days INTEGER DEFAULT 14)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM {schema}.event_log
    WHERE received_at < NOW() - (retention_days || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION {schema}.cleanup_old_events(INTEGER) IS
    'Deletes event_log rows older than retention_days. '
    'Default retention: 14 days. Returns count of deleted rows.';

-- ============================================================================
-- COMBINED CLEANUP
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.cleanup_sensor_history(
    tick_retention_days INTEGER DEFAULT 30,
    run_key_retention_days INTEGER DEFAULT 30,
    event_retention_days INTEGER DEFAULT 14
)
RETURNS TABLE (
    ticks_deleted INTEGER,
    run_keys_deleted INTEGER,
    events_deleted INTEGER
) AS $$
BEGIN
    ticks_deleted := {schema}.cleanup_old_ticks(tick_retention_days);
    run_keys_deleted := {schema}.cleanup_old_run_keys(run_key_retention_days);
    events_deleted := {schema}.cleanup_old_events(event_retention_days);
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION {schema}.cleanup_sensor_history(INTEGER, INTEGER, INTEGER) IS
    'Combined cleanup function for all sensor history tables. '
    'Returns counts of deleted rows for each table.';

-- ============================================================================
-- STUCK TICK CLEANUP
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.cleanup_stuck_ticks(threshold_hours INTEGER DEFAULT 24)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    -- Mark STARTED ticks older than threshold as FAILED
    -- These are likely from crashed daemon instances
    UPDATE {schema}.sensor_tick
    SET
        status = 'failed',
        error_message = 'Tick abandoned (daemon crash recovery)',
        completed_at = NOW(),
        duration_ms = EXTRACT(EPOCH FROM (NOW() - started_at))::INTEGER * 1000
    WHERE status = 'started'
      AND started_at < NOW() - (threshold_hours || ' hours')::INTERVAL;

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION {schema}.cleanup_stuck_ticks(INTEGER) IS
    'Marks STARTED ticks older than threshold_hours as FAILED. '
    'Used for crash recovery when daemon restarts. '
    'Default threshold: 24 hours.';
