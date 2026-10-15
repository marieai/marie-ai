-- Observability schema for the ClickHouse "otel" database.
--
-- Copied from config/clickhouse/init-clickhouse.sql (everything after the user bootstrap);
-- keep the two in step. Every statement is idempotent, so the schema Job can apply it on
-- install and on every upgrade.

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

-- Store frequent log facets so their counts do not scan the complete attribute map.
ALTER TABLE otel.otel_logs
    ADD COLUMN IF NOT EXISTS event_type LowCardinality(String) MATERIALIZED LogAttributes['event.type'],
    ADD COLUMN IF NOT EXISTS job_tag LowCardinality(String) MATERIALIZED LogAttributes['job.tag'],
    ADD COLUMN IF NOT EXISTS event_source String MATERIALIZED LogAttributes['event.source'];

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
CREATE TABLE IF NOT EXISTS otel.otel_metrics_exp_histogram (
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
    JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 2, 'capacity') as total_capacity,
    JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 2, 'used') as used,
    JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 2, 'available') as available,
    JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 2, 'holder_count') as holders
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

-- ############### OpenInference Materialized Columns ###############
-- Fast OI queries without Map access on otel_traces

ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    oi_span_kind LowCardinality(String) MATERIALIZED SpanAttributes['openinference.span.kind'];
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    llm_system LowCardinality(String) MATERIALIZED SpanAttributes['llm.system'];
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    llm_model_name LowCardinality(String) MATERIALIZED SpanAttributes['llm.model_name'];
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    llm_token_prompt Nullable(Int64) MATERIALIZED toInt64OrNull(SpanAttributes['llm.token_count.prompt']);
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    llm_token_completion Nullable(Int64) MATERIALIZED toInt64OrNull(SpanAttributes['llm.token_count.completion']);
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    llm_token_total Nullable(Int64) MATERIALIZED toInt64OrNull(SpanAttributes['llm.token_count.total']);
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    llm_cost_total Nullable(Float64) MATERIALIZED toFloat64OrNull(SpanAttributes['llm.cost.total']);
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    llm_cost_prompt Nullable(Float64) MATERIALIZED toFloat64OrNull(SpanAttributes['llm.cost.prompt']);
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    llm_cost_completion Nullable(Float64) MATERIALIZED toFloat64OrNull(SpanAttributes['llm.cost.completion']);
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    marie_project_id LowCardinality(String) MATERIALIZED SpanAttributes['marie.project_id'];
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    marie_observation_type LowCardinality(String) MATERIALIZED SpanAttributes['marie.observation_type'];
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    oi_session_id String MATERIALIZED SpanAttributes['session.id'];
ALTER TABLE otel.otel_traces ADD COLUMN IF NOT EXISTS
    oi_user_id String MATERIALIZED SpanAttributes['user.id'];

-- Indexes for common OI query patterns
ALTER TABLE otel.otel_traces ADD INDEX IF NOT EXISTS
    idx_oi_span_kind oi_span_kind TYPE set(10) GRANULARITY 4;
ALTER TABLE otel.otel_traces ADD INDEX IF NOT EXISTS
    idx_marie_project_id marie_project_id TYPE bloom_filter(0.001) GRANULARITY 1;
ALTER TABLE otel.otel_traces ADD INDEX IF NOT EXISTS
    idx_llm_model_name llm_model_name TYPE set(50) GRANULARITY 4;
ALTER TABLE otel.otel_traces ADD INDEX IF NOT EXISTS
    idx_oi_session_id oi_session_id TYPE bloom_filter(0.001) GRANULARITY 1;

-- ############### Span Annotations ###############
-- Evaluations, feedback, scores (Phoenix-compatible annotation model)

CREATE TABLE IF NOT EXISTS otel.span_annotations (
    id UUID DEFAULT generateUUIDv4(),
    trace_id String,
    span_id String,
    project_id String,
    name String,
    label Nullable(String),
    score Nullable(Float64),
    explanation Nullable(String),
    metadata String DEFAULT '{}',
    annotator_kind LowCardinality(String) DEFAULT 'HUMAN',
    source LowCardinality(String) DEFAULT 'API',
    created_at DateTime64(3) DEFAULT now64(3),
    updated_at DateTime64(3) DEFAULT now64(3),

    INDEX idx_trace_id trace_id TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_span_id span_id TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_project_id project_id TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_name name TYPE set(100) GRANULARITY 4
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (project_id, trace_id, name, id)
TTL toDateTime(created_at) + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- ############### LLM Model Pricing ###############
-- Price rules: a regex over the model name a span reports, priced per 1M
-- tokens. Built-in rows come from a catalog sync (OpenRouter); custom rows
-- win on match. Rows are versioned; is_deleted=1 removes a rule.
CREATE TABLE IF NOT EXISTS otel.model_prices (
    id UUID,
    name String,
    match_pattern String,
    provider String DEFAULT '',
    start_date Nullable(DateTime64(3, 'UTC')),
    input_price Float64,
    output_price Float64,
    input_breakdown Map(String, Float64),
    output_breakdown Map(String, Float64),
    is_built_in UInt8 DEFAULT 0,
    source LowCardinality(String) DEFAULT '',
    source_model_id String DEFAULT '',
    context_length Nullable(UInt32),
    metadata String DEFAULT '{}',
    created_by String DEFAULT '',
    created_at DateTime64(3, 'UTC') DEFAULT now64(3),
    updated_at DateTime64(3, 'UTC') DEFAULT now64(3),
    version UInt64,
    is_deleted UInt8 DEFAULT 0
) ENGINE = ReplacingMergeTree(version, is_deleted)
ORDER BY id;

CREATE TABLE IF NOT EXISTS otel.model_price_sync_runs (
    id UUID DEFAULT generateUUIDv4(),
    source LowCardinality(String),
    status LowCardinality(String),
    started_at DateTime64(3, 'UTC'),
    completed_at DateTime64(3, 'UTC') DEFAULT now64(3),
    fetched_count UInt32 DEFAULT 0,
    inserted_count UInt32 DEFAULT 0,
    updated_count UInt32 DEFAULT 0,
    stale_count UInt32 DEFAULT 0,
    error_message String DEFAULT ''
) ENGINE = MergeTree
ORDER BY (source, started_at)
TTL toDateTime(started_at) + INTERVAL 180 DAY;

-- Model/provider pairs that reported tokens, kept small so price resolution
-- never scans otel_traces
CREATE TABLE IF NOT EXISTS otel.llm_model_names (
    model String,
    provider String,
    last_seen DateTime
) ENGINE = ReplacingMergeTree(last_seen)
ORDER BY (model, provider);

CREATE MATERIALIZED VIEW IF NOT EXISTS otel.llm_model_names_mv TO otel.llm_model_names AS
SELECT
    SpanAttributes['llm.model_name'] AS model,
    SpanAttributes['llm.provider'] AS provider,
    max(toDateTime(Timestamp)) AS last_seen
FROM otel.otel_traces
WHERE SpanAttributes['llm.model_name'] != ''
    AND (SpanAttributes['llm.token_count.prompt'] != '' OR SpanAttributes['llm.token_count.completion'] != '')
GROUP BY model, provider;

-- The rule pricing each observed model/provider: custom before built-in,
-- provider-scoped before generic, latest start date in effect, then name.
-- Token-type rates fall back to the base rate.
CREATE DICTIONARY IF NOT EXISTS otel.model_price_dict (
    model String,
    provider String,
    rule_id String,
    input_price Float64,
    output_price Float64,
    input_cache_read Float64,
    input_cache_write Float64,
    input_audio Float64,
    output_reasoning Float64,
    output_audio Float64
)
PRIMARY KEY model, provider
SOURCE(CLICKHOUSE(QUERY '
    SELECT
        model,
        provider,
        toString(best.1) AS rule_id,
        best.2 AS input_price,
        best.3 AS output_price,
        if(mapContains(best.4, ''cache_read''), best.4[''cache_read''], best.2) AS input_cache_read,
        if(mapContains(best.4, ''cache_write''), best.4[''cache_write''], best.2) AS input_cache_write,
        if(mapContains(best.4, ''audio''), best.4[''audio''], best.2) AS input_audio,
        if(mapContains(best.5, ''reasoning''), best.5[''reasoning''], best.3) AS output_reasoning,
        if(mapContains(best.5, ''audio''), best.5[''audio''], best.3) AS output_audio
    FROM (
        SELECT
            n.model AS model,
            n.provider AS provider,
            argMin(
                (p.id, p.input_price, p.output_price, p.input_breakdown, p.output_breakdown),
                (p.is_built_in, p.provider = '''', -toUnixTimestamp64Milli(ifNull(p.start_date, toDateTime64(0, 3))), p.name)
            ) AS best
        FROM (SELECT DISTINCT model, provider FROM otel.llm_model_names) AS n
        CROSS JOIN (SELECT * FROM otel.model_prices FINAL) AS p
        WHERE match(n.model, concat(''(?i)'', p.match_pattern))
            AND (p.provider = '''' OR lower(p.provider) = lower(n.provider))
            AND (p.start_date IS NULL OR p.start_date <= now64(3))
        GROUP BY n.model, n.provider
    )
'))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(MIN 30 MAX 60);

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
