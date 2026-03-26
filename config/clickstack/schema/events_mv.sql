-- Event Viewer Materialized Views
-- Pre-compute expensive extractions at insert time, not query time

-- 1. Normalized events view (fast filtering on pre-extracted columns)
CREATE MATERIALIZED VIEW IF NOT EXISTS otel.events_mv
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (source_type, event_type, timestamp)
TTL timestamp + INTERVAL 30 DAY
AS SELECT
    Timestamp as timestamp,
    ServiceName as service,
    Body as body,
    SeverityText as severity,

    -- Pre-extracted for O(1) filtering (no JSON parsing at query time)
    LogAttributes['event.id'] as event_id,
    LogAttributes['event.type'] as event_type,
    LogAttributes['event.source'] as event_source,
    LogAttributes['event.source_type'] as source_type,
    LogAttributes['job.id'] as job_id,
    LogAttributes['job.tag'] as job_tag,
    LogAttributes['status'] as status,
    LogAttributes['api_key'] as api_key,
    LogAttributes['payload.message'] as message,
    LogAttributes['payload.metadata'] as metadata_json

FROM otel.otel_logs
WHERE length(LogAttributes['event.id']) > 0;


-- 2. Event counts by type (for filter badges, dashboard widgets)
CREATE MATERIALIZED VIEW IF NOT EXISTS otel.event_counts_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMMDD(timestamp_hour)
ORDER BY (timestamp_hour, source_type, event_type, status)
TTL timestamp_hour + INTERVAL 7 DAY
AS SELECT
    toStartOfHour(Timestamp) as timestamp_hour,
    LogAttributes['event.source_type'] as source_type,
    LogAttributes['event.type'] as event_type,
    LogAttributes['status'] as status,
    count() as count
FROM otel.otel_logs
WHERE length(LogAttributes['event.id']) > 0
GROUP BY timestamp_hour, source_type, event_type, status;


-- 3. Capacity time series (for sparkline charts)
CREATE MATERIALIZED VIEW IF NOT EXISTS otel.capacity_ts_mv
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY timestamp
TTL timestamp + INTERVAL 7 DAY
AS SELECT
    Timestamp as timestamp,
    -- Extract capacity summary from nested JSON
    JSONExtractInt(
        JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'),
        1, 2, 'capacity'
    ) as total_capacity,
    JSONExtractInt(
        JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'),
        1, 2, 'used'
    ) as used,
    JSONExtractInt(
        JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'),
        1, 2, 'available'
    ) as available,
    JSONExtractInt(
        JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'),
        1, 2, 'holder_count'
    ) as holders
FROM otel.otel_logs
WHERE LogAttributes['job.tag'] = 'RESOURCE_EXECUTOR_UPDATED';


-- 4. Job lifecycle tracking (for job detail view)
CREATE MATERIALIZED VIEW IF NOT EXISTS otel.job_events_mv
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (job_id, timestamp)
TTL timestamp + INTERVAL 30 DAY
AS SELECT
    Timestamp as timestamp,
    LogAttributes['job.id'] as job_id,
    LogAttributes['job.tag'] as job_tag,
    LogAttributes['event.type'] as event_type,
    LogAttributes['status'] as status,
    LogAttributes['payload.message'] as message,
    Body as body
FROM otel.otel_logs
WHERE LogAttributes['event.source_type'] = 'job';


-- Indexes for common query patterns
ALTER TABLE otel.otel_logs
    ADD INDEX IF NOT EXISTS idx_source_type (LogAttributes['event.source_type'])
    TYPE bloom_filter GRANULARITY 4;

ALTER TABLE otel.otel_logs
    ADD INDEX IF NOT EXISTS idx_event_type (LogAttributes['event.type'])
    TYPE bloom_filter GRANULARITY 4;

ALTER TABLE otel.otel_logs
    ADD INDEX IF NOT EXISTS idx_job_id (LogAttributes['job.id'])
    TYPE bloom_filter GRANULARITY 4;

ALTER TABLE otel.otel_logs
    ADD INDEX IF NOT EXISTS idx_api_key (LogAttributes['api_key'])
    TYPE bloom_filter GRANULARITY 4;


-- Example queries using the MVs (millisecond response times on millions of rows)

-- Get recent events with filters
-- SELECT * FROM otel.events_mv
-- WHERE source_type = 'gateway'
--   AND timestamp > now() - INTERVAL 1 HOUR
-- ORDER BY timestamp DESC
-- LIMIT 50;

-- Get event counts for filter badges
-- SELECT source_type, event_type, status, sum(count) as total
-- FROM otel.event_counts_mv
-- WHERE timestamp_hour > now() - INTERVAL 24 HOUR
-- GROUP BY source_type, event_type, status;

-- Get capacity time series for chart
-- SELECT
--     toStartOfMinute(timestamp) as minute,
--     max(total_capacity) as capacity,
--     max(used) as used,
--     max(available) as available
-- FROM otel.capacity_ts_mv
-- WHERE timestamp > now() - INTERVAL 1 HOUR
-- GROUP BY minute
-- ORDER BY minute;

-- Get all events for a specific job
-- SELECT * FROM otel.job_events_mv
-- WHERE job_id = 'abc-123-def'
-- ORDER BY timestamp;
