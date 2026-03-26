-- ClickHouse User Bootstrap Script
-- Creates application users for Marie AI stack
-- This script runs on container initialization

-- Create otel database if not exists
CREATE DATABASE IF NOT EXISTS otel;

-- Create marie user for application access
CREATE USER IF NOT EXISTS marie
    IDENTIFIED BY 'marie123'
    SETTINGS PROFILE 'default';

-- Grant permissions to marie user
GRANT ALL ON otel.* TO marie;
GRANT ALL ON marie.* TO marie;
GRANT SHOW DATABASES ON *.* TO marie;
GRANT SHOW TABLES ON *.* TO marie;
GRANT CREATE DATABASE ON *.* TO marie;
GRANT CREATE TABLE ON *.* TO marie;

-- Grant access to system tables (for DataGrip/IDE introspection)
GRANT SELECT ON system.* TO marie;
-- ############### ClickStack Observability Schema ###############
--
-- Tables for storing observability data (logs, traces, metrics)
-- Compatible with OpenTelemetry Collector ClickHouse Exporter v0.114.0+
--
-- Usage:
--   docker exec -i marie-clickhouse clickhouse-client < config/clickstack/schema/observability.sql

-- Create database for observability data
CREATE DATABASE IF NOT EXISTS otel;

-- ############### Logs Table ###############
-- Stores application and system logs
CREATE TABLE IF NOT EXISTS otel.otel_logs (
    Timestamp DateTime64(9) CODEC(Delta, ZSTD(1)),
    TraceId String CODEC(ZSTD(1)),
    SpanId String CODEC(ZSTD(1)),
    TraceFlags UInt32 CODEC(ZSTD(1)),
    SeverityText LowCardinality(String) CODEC(ZSTD(1)),
    SeverityNumber Int32 CODEC(ZSTD(1)),
    ServiceName LowCardinality(String) CODEC(ZSTD(1)),
    Body String CODEC(ZSTD(1)),
    ResourceSchemaUrl String CODEC(ZSTD(1)),
    ResourceAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeSchemaUrl String CODEC(ZSTD(1)),
    ScopeName String CODEC(ZSTD(1)),
    ScopeVersion String CODEC(ZSTD(1)),
    ScopeAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeDroppedAttrCount UInt32 DEFAULT 0 CODEC(ZSTD(1)),
    LogAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),

    INDEX idx_trace_id TraceId TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_severity SeverityText TYPE set(25) GRANULARITY 4,
    INDEX idx_body Body TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 1
)
ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (ServiceName, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

-- ############### Traces Table ###############
-- Stores distributed traces and spans
CREATE TABLE IF NOT EXISTS otel.otel_traces (
    Timestamp DateTime64(9) CODEC(Delta, ZSTD(1)),
    TraceId String CODEC(ZSTD(1)),
    SpanId String CODEC(ZSTD(1)),
    ParentSpanId String CODEC(ZSTD(1)),
    TraceState String CODEC(ZSTD(1)),
    SpanName LowCardinality(String) CODEC(ZSTD(1)),
    SpanKind LowCardinality(String) CODEC(ZSTD(1)),
    ServiceName LowCardinality(String) CODEC(ZSTD(1)),
    ResourceAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ResourceSchemaUrl String CODEC(ZSTD(1)),
    ScopeName String CODEC(ZSTD(1)),
    ScopeVersion String CODEC(ZSTD(1)),
    ScopeAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeDroppedAttrCount UInt32 DEFAULT 0 CODEC(ZSTD(1)),
    SpanAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    Duration Int64 CODEC(ZSTD(1)),
    StatusCode LowCardinality(String) CODEC(ZSTD(1)),
    StatusMessage String CODEC(ZSTD(1)),
    Events Nested (
        Timestamp DateTime64(9),
        Name LowCardinality(String),
        Attributes Map(LowCardinality(String), String)
    ) CODEC(ZSTD(1)),
    Links Nested (
        TraceId String,
        SpanId String,
        TraceState String,
        Attributes Map(LowCardinality(String), String)
    ) CODEC(ZSTD(1)),

    INDEX idx_trace_id TraceId TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_span_name SpanName TYPE set(100) GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (ServiceName, Timestamp, TraceId)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

-- ############### Metrics Tables ###############
-- Gauge metrics (point-in-time values)
CREATE TABLE IF NOT EXISTS otel.otel_metrics_gauge (
    ResourceAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ResourceSchemaUrl String CODEC(ZSTD(1)),
    ScopeName String CODEC(ZSTD(1)),
    ScopeVersion String CODEC(ZSTD(1)),
    ScopeAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeDroppedAttrCount UInt32 DEFAULT 0 CODEC(ZSTD(1)),
    ScopeSchemaUrl String CODEC(ZSTD(1)),
    ServiceName LowCardinality(String) CODEC(ZSTD(1)),
    MetricName String CODEC(ZSTD(1)),
    MetricDescription String CODEC(ZSTD(1)),
    MetricUnit String CODEC(ZSTD(1)),
    Attributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    StartTimeUnix DateTime64(9) CODEC(Delta, ZSTD(1)),
    TimeUnix DateTime64(9) CODEC(Delta, ZSTD(1)),
    Value Float64 CODEC(ZSTD(1)),
    Flags UInt32 CODEC(ZSTD(1)),
    Exemplars Nested (
        FilteredAttributes Map(LowCardinality(String), String),
        TimeUnix DateTime64(9),
        Value Float64,
        SpanId String,
        TraceId String
    ) CODEC(ZSTD(1))
)
ENGINE = MergeTree()
PARTITION BY toDate(TimeUnix)
ORDER BY (ServiceName, MetricName, Attributes, TimeUnix)
TTL toDateTime(TimeUnix) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

-- Sum metrics (cumulative or delta counters)
CREATE TABLE IF NOT EXISTS otel.otel_metrics_sum (
    ResourceAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ResourceSchemaUrl String CODEC(ZSTD(1)),
    ScopeName String CODEC(ZSTD(1)),
    ScopeVersion String CODEC(ZSTD(1)),
    ScopeAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeDroppedAttrCount UInt32 DEFAULT 0 CODEC(ZSTD(1)),
    ScopeSchemaUrl String CODEC(ZSTD(1)),
    ServiceName LowCardinality(String) CODEC(ZSTD(1)),
    MetricName String CODEC(ZSTD(1)),
    MetricDescription String CODEC(ZSTD(1)),
    MetricUnit String CODEC(ZSTD(1)),
    Attributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    StartTimeUnix DateTime64(9) CODEC(Delta, ZSTD(1)),
    TimeUnix DateTime64(9) CODEC(Delta, ZSTD(1)),
    Value Float64 CODEC(ZSTD(1)),
    Flags UInt32 CODEC(ZSTD(1)),
    AggregationTemporality Int32 CODEC(ZSTD(1)),
    IsMonotonic Bool CODEC(ZSTD(1)),
    Exemplars Nested (
        FilteredAttributes Map(LowCardinality(String), String),
        TimeUnix DateTime64(9),
        Value Float64,
        SpanId String,
        TraceId String
    ) CODEC(ZSTD(1))
)
ENGINE = MergeTree()
PARTITION BY toDate(TimeUnix)
ORDER BY (ServiceName, MetricName, Attributes, TimeUnix)
TTL toDateTime(TimeUnix) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

-- Histogram metrics
CREATE TABLE IF NOT EXISTS otel.otel_metrics_histogram (
    ResourceAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ResourceSchemaUrl String CODEC(ZSTD(1)),
    ScopeName String CODEC(ZSTD(1)),
    ScopeVersion String CODEC(ZSTD(1)),
    ScopeAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeDroppedAttrCount UInt32 DEFAULT 0 CODEC(ZSTD(1)),
    ScopeSchemaUrl String CODEC(ZSTD(1)),
    ServiceName LowCardinality(String) CODEC(ZSTD(1)),
    MetricName String CODEC(ZSTD(1)),
    MetricDescription String CODEC(ZSTD(1)),
    MetricUnit String CODEC(ZSTD(1)),
    Attributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    StartTimeUnix DateTime64(9) CODEC(Delta, ZSTD(1)),
    TimeUnix DateTime64(9) CODEC(Delta, ZSTD(1)),
    Count UInt64 CODEC(ZSTD(1)),
    Sum Float64 CODEC(ZSTD(1)),
    BucketCounts Array(UInt64) CODEC(ZSTD(1)),
    ExplicitBounds Array(Float64) CODEC(ZSTD(1)),
    Flags UInt32 CODEC(ZSTD(1)),
    Min Float64 CODEC(ZSTD(1)),
    Max Float64 CODEC(ZSTD(1)),
    AggregationTemporality Int32 CODEC(ZSTD(1)),
    Exemplars Nested (
        FilteredAttributes Map(LowCardinality(String), String),
        TimeUnix DateTime64(9),
        Value Float64,
        SpanId String,
        TraceId String
    ) CODEC(ZSTD(1))
)
ENGINE = MergeTree()
PARTITION BY toDate(TimeUnix)
ORDER BY (ServiceName, MetricName, Attributes, TimeUnix)
TTL toDateTime(TimeUnix) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

-- Exponential Histogram metrics (for high-cardinality distributions)
CREATE TABLE IF NOT EXISTS otel.otel_metrics_exponential_histogram (
    ResourceAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ResourceSchemaUrl String CODEC(ZSTD(1)),
    ScopeName String CODEC(ZSTD(1)),
    ScopeVersion String CODEC(ZSTD(1)),
    ScopeAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeDroppedAttrCount UInt32 DEFAULT 0 CODEC(ZSTD(1)),
    ScopeSchemaUrl String CODEC(ZSTD(1)),
    ServiceName LowCardinality(String) CODEC(ZSTD(1)),
    MetricName String CODEC(ZSTD(1)),
    MetricDescription String CODEC(ZSTD(1)),
    MetricUnit String CODEC(ZSTD(1)),
    Attributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    StartTimeUnix DateTime64(9) CODEC(Delta, ZSTD(1)),
    TimeUnix DateTime64(9) CODEC(Delta, ZSTD(1)),
    Count UInt64 CODEC(ZSTD(1)),
    Sum Float64 CODEC(ZSTD(1)),
    Scale Int32 CODEC(ZSTD(1)),
    ZeroCount UInt64 CODEC(ZSTD(1)),
    PositiveOffset Int32 CODEC(ZSTD(1)),
    PositiveBucketCounts Array(UInt64) CODEC(ZSTD(1)),
    NegativeOffset Int32 CODEC(ZSTD(1)),
    NegativeBucketCounts Array(UInt64) CODEC(ZSTD(1)),
    Flags UInt32 CODEC(ZSTD(1)),
    Min Float64 CODEC(ZSTD(1)),
    Max Float64 CODEC(ZSTD(1)),
    AggregationTemporality Int32 CODEC(ZSTD(1)),
    Exemplars Nested (
        FilteredAttributes Map(LowCardinality(String), String),
        TimeUnix DateTime64(9),
        Value Float64,
        SpanId String,
        TraceId String
    ) CODEC(ZSTD(1))
)
ENGINE = MergeTree()
PARTITION BY toDate(TimeUnix)
ORDER BY (ServiceName, MetricName, Attributes, TimeUnix)
TTL toDateTime(TimeUnix) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

-- Summary metrics
CREATE TABLE IF NOT EXISTS otel.otel_metrics_summary (
    ResourceAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ResourceSchemaUrl String CODEC(ZSTD(1)),
    ScopeName String CODEC(ZSTD(1)),
    ScopeVersion String CODEC(ZSTD(1)),
    ScopeAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeDroppedAttrCount UInt32 DEFAULT 0 CODEC(ZSTD(1)),
    ScopeSchemaUrl String CODEC(ZSTD(1)),
    ServiceName LowCardinality(String) CODEC(ZSTD(1)),
    MetricName String CODEC(ZSTD(1)),
    MetricDescription String CODEC(ZSTD(1)),
    MetricUnit String CODEC(ZSTD(1)),
    Attributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    StartTimeUnix DateTime64(9) CODEC(Delta, ZSTD(1)),
    TimeUnix DateTime64(9) CODEC(Delta, ZSTD(1)),
    Count UInt64 CODEC(ZSTD(1)),
    Sum Float64 CODEC(ZSTD(1)),
    ValueAtQuantiles Nested (
        Quantile Float64,
        Value Float64
    ) CODEC(ZSTD(1)),
    Flags UInt32 CODEC(ZSTD(1))
)
ENGINE = MergeTree()
PARTITION BY toDate(TimeUnix)
ORDER BY (ServiceName, MetricName, Attributes, TimeUnix)
TTL toDateTime(TimeUnix) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

-- ############### Materialized Views for Error Monitoring ###############

-- Error logs view for quick error analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS otel.error_logs_mv
ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (ServiceName, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY
AS SELECT
    Timestamp,
    ServiceName,
    SeverityText,
    Body,
    TraceId,
    SpanId,
    LogAttributes
FROM otel.otel_logs
WHERE SeverityText IN ('ERROR', 'FATAL', 'CRITICAL');

-- Error traces view for failed spans
CREATE MATERIALIZED VIEW IF NOT EXISTS otel.error_traces_mv
ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (ServiceName, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY
AS SELECT
    Timestamp,
    ServiceName,
    SpanName,
    TraceId,
    SpanId,
    Duration,
    StatusCode,
    StatusMessage,
    SpanAttributes
FROM otel.otel_traces
WHERE StatusCode = 'ERROR';

-- ############### Event Viewer Materialized Views ###############
-- Pre-compute expensive extractions at insert time, not query time

-- Normalized events view (fast filtering on pre-extracted columns)
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

-- Event counts by type (for filter badges, dashboard widgets)
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

-- Capacity time series (for sparkline charts)
CREATE MATERIALIZED VIEW IF NOT EXISTS otel.capacity_ts_mv
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY timestamp
TTL timestamp + INTERVAL 7 DAY
AS SELECT
    Timestamp as timestamp,
    JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 1, 2, 'capacity') as total_capacity,
    JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 1, 2, 'used') as used,
    JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 1, 2, 'available') as available,
    JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 1, 2, 'holder_count') as holders
FROM otel.otel_logs
WHERE LogAttributes['job.tag'] = 'RESOURCE_EXECUTOR_UPDATED';

-- Job lifecycle tracking (for job detail view)
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

-- ############### Event Viewer Indexes ###############
-- Bloom filters for fast cardinality filtering on Map keys

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

-- ############### Useful Queries ###############
--
-- Recent errors:
--   SELECT * FROM otel.error_logs_mv ORDER BY Timestamp DESC LIMIT 100;
--
-- Error count by service (last hour):
--   SELECT ServiceName, count() as error_count
--   FROM otel.error_logs_mv
--   WHERE Timestamp > now() - INTERVAL 1 HOUR
--   GROUP BY ServiceName
--   ORDER BY error_count DESC;
--
-- Slowest traces (last hour):
--   SELECT ServiceName, SpanName, TraceId, Duration/1000000 as duration_ms
--   FROM otel.otel_traces
--   WHERE Timestamp > now() - INTERVAL 1 HOUR
--   ORDER BY Duration DESC
--   LIMIT 20;
--
-- Gateway request metrics (last hour):
--   SELECT
--     toStartOfMinute(TimeUnix) as minute,
--     sum(Count) as requests,
--     avg(Sum / Count) * 1000 as avg_latency_ms
--   FROM otel.otel_metrics_histogram
--   WHERE MetricName = 'marie_gateway_request_seconds'
--     AND TimeUnix > now() - INTERVAL 1 HOUR
--   GROUP BY minute
--   ORDER BY minute DESC;
--
-- Event viewer queries (use events_mv for fast filtering):
--   SELECT * FROM otel.events_mv
--   WHERE source_type = 'gateway' AND timestamp > now() - INTERVAL 1 HOUR
--   ORDER BY timestamp DESC LIMIT 50;
--
--   SELECT source_type, event_type, sum(count) as total
--   FROM otel.event_counts_mv
--   WHERE timestamp_hour > now() - INTERVAL 24 HOUR
--   GROUP BY source_type, event_type;
