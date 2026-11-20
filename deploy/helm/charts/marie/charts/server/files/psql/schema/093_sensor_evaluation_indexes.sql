-- Match Studio's existing index for installations bootstrapped from SQL files.
CREATE INDEX IF NOT EXISTS sensor_tick_sensor_id_started_at_idx
ON {schema}.sensor_tick (sensor_id, started_at DESC);
