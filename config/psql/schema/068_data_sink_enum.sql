-- 068_data_sink_enum.sql
-- Adds 'data_sink' to both sensor-related enums.
-- trigger_type is the type of marie_scheduler.sensor.sensor_type (052_sensor_tables.sql:19);
-- sensor_type is used by event_log. Python SensorType.DATA_SINK already exists (marie/sensors/types.py:27).
ALTER TYPE marie_scheduler.trigger_type ADD VALUE IF NOT EXISTS 'data_sink';
ALTER TYPE marie_scheduler.sensor_type ADD VALUE IF NOT EXISTS 'data_sink';
