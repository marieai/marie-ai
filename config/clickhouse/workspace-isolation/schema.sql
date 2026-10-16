-- Additive preparation only. No legacy copy, ingestion switch or tenant grants.

CREATE DATABASE IF NOT EXISTS telemetry;

CREATE DATABASE IF NOT EXISTS workspace_query;

CREATE TABLE IF NOT EXISTS telemetry.spans (
    organization_id String DEFAULT '',
    workspace_id String DEFAULT '',
    identity_state Enum8('unattributed' = 0, 'verified' = 1, 'platform' = 2) DEFAULT 'unattributed',
    identity_version UInt8 DEFAULT 0,
    CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND toUUIDOrNull(workspace_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND workspace_id = toString(toUUIDOrNull(workspace_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND workspace_id != '00000000-0000-0000-0000-000000000000'),

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
) ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, Timestamp, TraceId, SpanId)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS telemetry.logs (
    organization_id String DEFAULT '',
    workspace_id String DEFAULT '',
    identity_state Enum8('unattributed' = 0, 'verified' = 1, 'platform' = 2) DEFAULT 'unattributed',
    identity_version UInt8 DEFAULT 0,
    CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND toUUIDOrNull(workspace_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND workspace_id = toString(toUUIDOrNull(workspace_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND workspace_id != '00000000-0000-0000-0000-000000000000'),

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
) ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, Timestamp, TraceId, SpanId)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS telemetry.metrics_gauge (
    organization_id String DEFAULT '',
    workspace_id String DEFAULT '',
    identity_state Enum8('unattributed' = 0, 'verified' = 1, 'platform' = 2) DEFAULT 'unattributed',
    identity_version UInt8 DEFAULT 0,
    CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND toUUIDOrNull(workspace_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND workspace_id = toString(toUUIDOrNull(workspace_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND workspace_id != '00000000-0000-0000-0000-000000000000'),

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
) ENGINE = MergeTree()
PARTITION BY toDate(TimeUnix)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, MetricName, Attributes, TimeUnix)
TTL toDateTime(TimeUnix) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS telemetry.metrics_sum (
    organization_id String DEFAULT '',
    workspace_id String DEFAULT '',
    identity_state Enum8('unattributed' = 0, 'verified' = 1, 'platform' = 2) DEFAULT 'unattributed',
    identity_version UInt8 DEFAULT 0,
    CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND toUUIDOrNull(workspace_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND workspace_id = toString(toUUIDOrNull(workspace_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND workspace_id != '00000000-0000-0000-0000-000000000000'),

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
) ENGINE = MergeTree()
PARTITION BY toDate(TimeUnix)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, MetricName, Attributes, TimeUnix)
TTL toDateTime(TimeUnix) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS telemetry.metrics_histogram (
    organization_id String DEFAULT '',
    workspace_id String DEFAULT '',
    identity_state Enum8('unattributed' = 0, 'verified' = 1, 'platform' = 2) DEFAULT 'unattributed',
    identity_version UInt8 DEFAULT 0,
    CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND toUUIDOrNull(workspace_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND workspace_id = toString(toUUIDOrNull(workspace_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND workspace_id != '00000000-0000-0000-0000-000000000000'),

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
) ENGINE = MergeTree()
PARTITION BY toDate(TimeUnix)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, MetricName, Attributes, TimeUnix)
TTL toDateTime(TimeUnix) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS telemetry.metrics_exponential_histogram (
    organization_id String DEFAULT '',
    workspace_id String DEFAULT '',
    identity_state Enum8('unattributed' = 0, 'verified' = 1, 'platform' = 2) DEFAULT 'unattributed',
    identity_version UInt8 DEFAULT 0,
    CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND toUUIDOrNull(workspace_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND workspace_id = toString(toUUIDOrNull(workspace_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND workspace_id != '00000000-0000-0000-0000-000000000000'),

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
) ENGINE = MergeTree()
PARTITION BY toDate(TimeUnix)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, MetricName, Attributes, TimeUnix)
TTL toDateTime(TimeUnix) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS telemetry.metrics_summary (
    organization_id String DEFAULT '',
    workspace_id String DEFAULT '',
    identity_state Enum8('unattributed' = 0, 'verified' = 1, 'platform' = 2) DEFAULT 'unattributed',
    identity_version UInt8 DEFAULT 0,
    CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND toUUIDOrNull(workspace_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND workspace_id = toString(toUUIDOrNull(workspace_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND workspace_id != '00000000-0000-0000-0000-000000000000'),

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
) ENGINE = MergeTree()
PARTITION BY toDate(TimeUnix)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, MetricName, Attributes, TimeUnix)
TTL toDateTime(TimeUnix) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    oi_span_kind LowCardinality(String) MATERIALIZED SpanAttributes['openinference.span.kind'];

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    llm_system LowCardinality(String) MATERIALIZED SpanAttributes['llm.system'];

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    llm_model_name LowCardinality(String) MATERIALIZED SpanAttributes['llm.model_name'];

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    llm_token_prompt Nullable(Int64) MATERIALIZED toInt64OrNull(SpanAttributes['llm.token_count.prompt']);

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    llm_token_completion Nullable(Int64) MATERIALIZED toInt64OrNull(SpanAttributes['llm.token_count.completion']);

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    llm_token_total Nullable(Int64) MATERIALIZED toInt64OrNull(SpanAttributes['llm.token_count.total']);

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    llm_cost_total Nullable(Float64) MATERIALIZED toFloat64OrNull(SpanAttributes['llm.cost.total']);

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    llm_cost_prompt Nullable(Float64) MATERIALIZED toFloat64OrNull(SpanAttributes['llm.cost.prompt']);

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    llm_cost_completion Nullable(Float64) MATERIALIZED toFloat64OrNull(SpanAttributes['llm.cost.completion']);

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    marie_project_id LowCardinality(String) MATERIALIZED SpanAttributes['marie.project_id'];

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    marie_observation_type LowCardinality(String) MATERIALIZED SpanAttributes['marie.observation_type'];

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    oi_session_id String MATERIALIZED SpanAttributes['session.id'];

ALTER TABLE telemetry.spans ADD COLUMN IF NOT EXISTS
    oi_user_id String MATERIALIZED SpanAttributes['user.id'];

CREATE TABLE IF NOT EXISTS telemetry.annotations (
    organization_id String DEFAULT '',
    workspace_id String DEFAULT '',
    identity_state Enum8('unattributed' = 0, 'verified' = 1, 'platform' = 2) DEFAULT 'unattributed',
    identity_version UInt8 DEFAULT 0,
    CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND toUUIDOrNull(workspace_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND workspace_id = toString(toUUIDOrNull(workspace_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND workspace_id != '00000000-0000-0000-0000-000000000000'),
    id UUID DEFAULT generateUUIDv4(),
    trace_id String,
    span_id String,
    name String,
    label Nullable(String),
    score Nullable(Float64),
    explanation Nullable(String),
    metadata String DEFAULT '{}',
    annotator_kind LowCardinality(String) DEFAULT 'HUMAN',
    source LowCardinality(String) DEFAULT 'API',
    created_at DateTime64(3) DEFAULT now64(3),
    updated_at DateTime64(3) DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, trace_id, span_id, name, id)
TTL toDateTime(created_at) + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS telemetry.model_names (
    organization_id String DEFAULT '',
    workspace_id String DEFAULT '',
    identity_state Enum8('unattributed' = 0, 'verified' = 1, 'platform' = 2) DEFAULT 'unattributed',
    identity_version UInt8 DEFAULT 0,
    CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND toUUIDOrNull(workspace_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND workspace_id = toString(toUUIDOrNull(workspace_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND workspace_id != '00000000-0000-0000-0000-000000000000'),
    model String,
    provider String,
    last_seen DateTime
) ENGINE = ReplacingMergeTree(last_seen)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, model, provider)
;

CREATE TABLE IF NOT EXISTS telemetry.event_counts (
    organization_id String DEFAULT '',
    workspace_id String DEFAULT '',
    identity_state Enum8('unattributed' = 0, 'verified' = 1, 'platform' = 2) DEFAULT 'unattributed',
    identity_version UInt8 DEFAULT 0,
    CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR
        (identity_version = 1 AND toUUIDOrNull(organization_id) IS NOT NULL
         AND toUUIDOrNull(workspace_id) IS NOT NULL
         AND organization_id = toString(toUUIDOrNull(organization_id))
         AND workspace_id = toString(toUUIDOrNull(workspace_id))
         AND organization_id != '00000000-0000-0000-0000-000000000000'
         AND workspace_id != '00000000-0000-0000-0000-000000000000'),
    timestamp_hour DateTime,
    source_type String,
    event_type String,
    status String,
    count UInt64
) ENGINE = SummingMergeTree(count)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, timestamp_hour, source_type, event_type, status)
TTL timestamp_hour + INTERVAL 7 DAY;

CREATE TABLE IF NOT EXISTS telemetry.error_logs
ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY
AS SELECT organization_id, workspace_id, identity_state, identity_version, Timestamp, ServiceName, SeverityText, Body, TraceId, SpanId, LogAttributes FROM telemetry.logs WHERE 0;

CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry.error_logs_ingest TO telemetry.error_logs
AS SELECT organization_id, workspace_id, identity_state, identity_version, Timestamp, ServiceName, SeverityText, Body, TraceId, SpanId, LogAttributes FROM telemetry.logs WHERE SeverityText IN ('ERROR', 'FATAL', 'CRITICAL');

CREATE TABLE IF NOT EXISTS telemetry.error_spans
ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY
AS SELECT organization_id, workspace_id, identity_state, identity_version, Timestamp, ServiceName, SpanName, TraceId, SpanId, Duration, StatusCode, StatusMessage, SpanAttributes FROM telemetry.spans WHERE 0;

CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry.error_spans_ingest TO telemetry.error_spans
AS SELECT organization_id, workspace_id, identity_state, identity_version, Timestamp, ServiceName, SpanName, TraceId, SpanId, Duration, StatusCode, StatusMessage, SpanAttributes FROM telemetry.spans WHERE StatusCode = 'ERROR';

CREATE TABLE IF NOT EXISTS telemetry.events
ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY
AS SELECT organization_id, workspace_id, identity_state, identity_version, Timestamp, ServiceName, Body, SeverityText, LogAttributes['event.id'] AS event_id, LogAttributes['event.type'] AS event_type, LogAttributes['event.source'] AS event_source, LogAttributes['event.source_type'] AS source_type, LogAttributes['job.id'] AS job_id, LogAttributes['job.tag'] AS job_tag, LogAttributes['status'] AS status FROM telemetry.logs WHERE 0;

CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry.events_ingest TO telemetry.events
AS SELECT organization_id, workspace_id, identity_state, identity_version, Timestamp, ServiceName, Body, SeverityText, LogAttributes['event.id'] AS event_id, LogAttributes['event.type'] AS event_type, LogAttributes['event.source'] AS event_source, LogAttributes['event.source_type'] AS source_type, LogAttributes['job.id'] AS job_id, LogAttributes['job.tag'] AS job_tag, LogAttributes['status'] AS status FROM telemetry.logs WHERE LogAttributes['event.id'] != '';

CREATE TABLE IF NOT EXISTS telemetry.job_events
ENGINE = MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (organization_id, workspace_id, identity_state, identity_version, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY
AS SELECT organization_id, workspace_id, identity_state, identity_version, Timestamp, LogAttributes['job.id'] AS job_id, LogAttributes['job.tag'] AS job_tag, LogAttributes['event.type'] AS event_type, LogAttributes['status'] AS status, LogAttributes['payload.message'] AS message, Body FROM telemetry.logs WHERE 0;

CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry.job_events_ingest TO telemetry.job_events
AS SELECT organization_id, workspace_id, identity_state, identity_version, Timestamp, LogAttributes['job.id'] AS job_id, LogAttributes['job.tag'] AS job_tag, LogAttributes['event.type'] AS event_type, LogAttributes['status'] AS status, LogAttributes['payload.message'] AS message, Body FROM telemetry.logs WHERE LogAttributes['event.source_type'] = 'job';

CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry.event_counts_ingest TO telemetry.event_counts AS
SELECT organization_id, workspace_id, identity_state, identity_version,
       toStartOfHour(Timestamp) AS timestamp_hour,
       LogAttributes['event.source_type'] AS source_type,
       LogAttributes['event.type'] AS event_type, LogAttributes['status'] AS status, count() AS count
FROM telemetry.logs WHERE LogAttributes['event.id'] != ''
GROUP BY organization_id, workspace_id, identity_state, identity_version, timestamp_hour, source_type, event_type, status;

CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry.model_names_ingest TO telemetry.model_names AS
SELECT organization_id, workspace_id, identity_state, identity_version,
       llm_model_name AS model, SpanAttributes['llm.provider'] AS provider, max(toDateTime(Timestamp)) AS last_seen
FROM telemetry.spans WHERE llm_model_name != '' AND (llm_token_prompt IS NOT NULL OR llm_token_completion IS NOT NULL)
GROUP BY organization_id, workspace_id, identity_state, identity_version, model, provider;

CREATE VIEW IF NOT EXISTS workspace_query.spans SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `Timestamp`, `TraceId`, `SpanId`, `ParentSpanId`, `TraceState`, `SpanName`, `SpanKind`, `ServiceName`, `ResourceAttributes`, `ResourceSchemaUrl`, `ScopeName`, `ScopeVersion`, `ScopeAttributes`, `ScopeDroppedAttrCount`, `SpanAttributes`, `Duration`, `StatusCode`, `StatusMessage`, `Events.Timestamp`, `Events.Name`, `Events.Attributes`, `Links.TraceId`, `Links.SpanId`, `Links.TraceState`, `Links.Attributes`, `oi_span_kind`, `llm_system`, `llm_model_name`, `llm_token_prompt`, `llm_token_completion`, `llm_token_total`, `llm_cost_total`, `llm_cost_prompt`, `llm_cost_completion`, `marie_project_id`, `marie_observation_type`, `oi_session_id`, `oi_user_id`, llm_cost_total AS cost_total, llm_cost_prompt AS cost_prompt, llm_cost_completion AS cost_completion, Duration / 1000000 AS latency_ms FROM telemetry.spans;

CREATE VIEW IF NOT EXISTS workspace_query.logs SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `Timestamp`, `TraceId`, `SpanId`, `TraceFlags`, `SeverityText`, `SeverityNumber`, `ServiceName`, `Body`, `ResourceSchemaUrl`, `ResourceAttributes`, `ScopeSchemaUrl`, `ScopeName`, `ScopeVersion`, `ScopeAttributes`, `ScopeDroppedAttrCount`, `LogAttributes` FROM telemetry.logs;

CREATE VIEW IF NOT EXISTS workspace_query.metrics_gauge SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `ResourceAttributes`, `ResourceSchemaUrl`, `ScopeName`, `ScopeVersion`, `ScopeAttributes`, `ScopeDroppedAttrCount`, `ScopeSchemaUrl`, `ServiceName`, `MetricName`, `MetricDescription`, `MetricUnit`, `Attributes`, `StartTimeUnix`, `TimeUnix`, `Value`, `Flags`, `Exemplars.FilteredAttributes`, `Exemplars.TimeUnix`, `Exemplars.Value`, `Exemplars.SpanId`, `Exemplars.TraceId` FROM telemetry.metrics_gauge;

CREATE VIEW IF NOT EXISTS workspace_query.metrics_sum SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `ResourceAttributes`, `ResourceSchemaUrl`, `ScopeName`, `ScopeVersion`, `ScopeAttributes`, `ScopeDroppedAttrCount`, `ScopeSchemaUrl`, `ServiceName`, `MetricName`, `MetricDescription`, `MetricUnit`, `Attributes`, `StartTimeUnix`, `TimeUnix`, `Value`, `Flags`, `AggregationTemporality`, `IsMonotonic`, `Exemplars.FilteredAttributes`, `Exemplars.TimeUnix`, `Exemplars.Value`, `Exemplars.SpanId`, `Exemplars.TraceId` FROM telemetry.metrics_sum;

CREATE VIEW IF NOT EXISTS workspace_query.metrics_histogram SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `ResourceAttributes`, `ResourceSchemaUrl`, `ScopeName`, `ScopeVersion`, `ScopeAttributes`, `ScopeDroppedAttrCount`, `ScopeSchemaUrl`, `ServiceName`, `MetricName`, `MetricDescription`, `MetricUnit`, `Attributes`, `StartTimeUnix`, `TimeUnix`, `Count`, `Sum`, `BucketCounts`, `ExplicitBounds`, `Flags`, `Min`, `Max`, `AggregationTemporality`, `Exemplars.FilteredAttributes`, `Exemplars.TimeUnix`, `Exemplars.Value`, `Exemplars.SpanId`, `Exemplars.TraceId` FROM telemetry.metrics_histogram;

CREATE VIEW IF NOT EXISTS workspace_query.metrics_exponential_histogram SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `ResourceAttributes`, `ResourceSchemaUrl`, `ScopeName`, `ScopeVersion`, `ScopeAttributes`, `ScopeDroppedAttrCount`, `ScopeSchemaUrl`, `ServiceName`, `MetricName`, `MetricDescription`, `MetricUnit`, `Attributes`, `StartTimeUnix`, `TimeUnix`, `Count`, `Sum`, `Scale`, `ZeroCount`, `PositiveOffset`, `PositiveBucketCounts`, `NegativeOffset`, `NegativeBucketCounts`, `Flags`, `Min`, `Max`, `AggregationTemporality`, `Exemplars.FilteredAttributes`, `Exemplars.TimeUnix`, `Exemplars.Value`, `Exemplars.SpanId`, `Exemplars.TraceId` FROM telemetry.metrics_exponential_histogram;

CREATE VIEW IF NOT EXISTS workspace_query.metrics_summary SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `ResourceAttributes`, `ResourceSchemaUrl`, `ScopeName`, `ScopeVersion`, `ScopeAttributes`, `ScopeDroppedAttrCount`, `ScopeSchemaUrl`, `ServiceName`, `MetricName`, `MetricDescription`, `MetricUnit`, `Attributes`, `StartTimeUnix`, `TimeUnix`, `Count`, `Sum`, `ValueAtQuantiles.Quantile`, `ValueAtQuantiles.Value`, `Flags` FROM telemetry.metrics_summary;

CREATE VIEW IF NOT EXISTS workspace_query.annotations SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `id`, `trace_id`, `span_id`, `name`, `label`, `score`, `explanation`, `metadata`, `annotator_kind`, `source`, `created_at`, `updated_at` FROM telemetry.annotations FINAL;

CREATE VIEW IF NOT EXISTS workspace_query.model_names SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `model`, `provider`, `last_seen` FROM telemetry.model_names FINAL;

CREATE VIEW IF NOT EXISTS workspace_query.event_counts SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `timestamp_hour`, `source_type`, `event_type`, `status`, `count` FROM telemetry.event_counts;

CREATE VIEW IF NOT EXISTS workspace_query.error_logs SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `Timestamp`, `ServiceName`, `SeverityText`, `Body`, `TraceId`, `SpanId`, `LogAttributes` FROM telemetry.error_logs;

CREATE VIEW IF NOT EXISTS workspace_query.error_spans SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `Timestamp`, `ServiceName`, `SpanName`, `TraceId`, `SpanId`, `Duration`, `StatusCode`, `StatusMessage`, `SpanAttributes` FROM telemetry.error_spans;

CREATE VIEW IF NOT EXISTS workspace_query.events SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `Timestamp`, `ServiceName`, `Body`, `SeverityText`, `event_id`, `event_type`, `event_source`, `source_type`, `job_id`, `job_tag`, `status` FROM telemetry.events;

CREATE VIEW IF NOT EXISTS workspace_query.job_events SQL SECURITY INVOKER AS SELECT `organization_id`, `workspace_id`, `identity_state`, `identity_version`, `Timestamp`, `job_id`, `job_tag`, `event_type`, `status`, `message`, `Body` FROM telemetry.job_events;

CREATE VIEW IF NOT EXISTS workspace_query.traces SQL SECURITY INVOKER AS SELECT * FROM workspace_query.spans WHERE ParentSpanId = '';
