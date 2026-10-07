-- Store frequent log facets so their counts do not scan the complete attribute map.
ALTER TABLE otel.otel_logs
    ADD COLUMN IF NOT EXISTS event_type LowCardinality(String) MATERIALIZED LogAttributes['event.type'],
    ADD COLUMN IF NOT EXISTS job_tag LowCardinality(String) MATERIALIZED LogAttributes['job.tag'],
    ADD COLUMN IF NOT EXISTS event_source String MATERIALIZED LogAttributes['event.source'];

-- Run once for existing installations; wait until existing parts have stored values.
ALTER TABLE otel.otel_logs
    MATERIALIZE COLUMN event_type,
    MATERIALIZE COLUMN job_tag,
    MATERIALIZE COLUMN event_source
    SETTINGS mutations_sync = 1;
