-- ClickHouse Schema for LLM Tracking
-- Database: marie_llm
--
-- This schema stores traces, observations, and scores for LLM observability.
-- Inspired by Langfuse's ClickHouse schema.
--
-- Apply with:
--   clickhouse-client --database marie_llm < config/clickhouse/schema/llm_tracking.sql

-- ============================================================
-- TRACES TABLE
-- Root container for LLM interactions
-- ============================================================
CREATE TABLE IF NOT EXISTS traces (
    -- Identifiers
    id UUID,
    project_id String,

    -- Core fields
    name String,
    user_id Nullable(String),
    session_id Nullable(String),

    -- Content (JSON strings for flexibility)
    input String DEFAULT '',
    output String DEFAULT '',
    metadata String DEFAULT '{}',

    -- Classification
    tags Array(String) DEFAULT [],
    release Nullable(String),
    version Nullable(String),

    -- Timestamps
    timestamp DateTime64(3),
    created_at DateTime64(3),
    updated_at DateTime64(3),
    event_ts DateTime64(3) DEFAULT now64(3),

    -- Soft delete flag
    is_deleted UInt8 DEFAULT 0,

    -- Indexes
    INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 1,
    INDEX idx_user_id user_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_session_id session_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_name name TYPE bloom_filter GRANULARITY 1

) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (project_id, timestamp, id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 365 DAY;


-- ============================================================
-- OBSERVATIONS TABLE
-- Spans, Generations, and Events within traces
-- ============================================================
CREATE TABLE IF NOT EXISTS observations (
    -- Identifiers
    id UUID,
    trace_id UUID,
    project_id String,
    parent_observation_id Nullable(UUID),

    -- Type: SPAN, GENERATION, EVENT
    type LowCardinality(String),
    name String,

    -- Timing
    start_time DateTime64(3),
    end_time Nullable(DateTime64(3)),
    completion_start_time Nullable(DateTime64(3)),  -- Time to first token

    -- Model info (for GENERATION type)
    model Nullable(String),
    model_parameters Nullable(String),  -- JSON

    -- Token usage
    input_tokens Nullable(Int64),
    output_tokens Nullable(Int64),
    total_tokens Nullable(Int64),
    usage_details String DEFAULT '{}',  -- JSON for extended usage (cached_tokens, etc.)

    -- Cost
    input_cost Nullable(Decimal64(8)),
    output_cost Nullable(Decimal64(8)),
    total_cost Nullable(Decimal64(8)),
    cost_details String DEFAULT '{}',  -- JSON for extended cost details

    -- Content
    input String DEFAULT '',
    output String DEFAULT '',
    metadata String DEFAULT '{}',

    -- Status
    level LowCardinality(String) DEFAULT 'DEFAULT',  -- DEBUG, DEFAULT, WARNING, ERROR
    status_message Nullable(String),

    -- Versioning
    version Nullable(String),

    -- Timestamps
    created_at DateTime64(3),
    updated_at DateTime64(3),
    event_ts DateTime64(3) DEFAULT now64(3),

    -- Soft delete
    is_deleted UInt8 DEFAULT 0,

    -- Indexes
    INDEX idx_trace_id trace_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_start_time start_time TYPE minmax GRANULARITY 1,
    INDEX idx_type type TYPE set(10) GRANULARITY 1,
    INDEX idx_model model TYPE bloom_filter GRANULARITY 1

) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (project_id, trace_id, start_time, id)
PARTITION BY toYYYYMM(start_time)
TTL start_time + INTERVAL 365 DAY;


-- ============================================================
-- SCORES TABLE
-- Evaluation scores for traces and observations
-- ============================================================
CREATE TABLE IF NOT EXISTS scores (
    -- Identifiers
    id UUID,
    trace_id UUID,
    observation_id Nullable(UUID),
    project_id String,

    -- Score info
    name String,
    value Float64,  -- Numeric representation
    string_value Nullable(String),  -- For categorical/string values
    data_type LowCardinality(String),  -- NUMERIC, CATEGORICAL, BOOLEAN
    source LowCardinality(String) DEFAULT 'API',  -- API, EVAL, ANNOTATION

    -- Details
    comment Nullable(String),
    metadata String DEFAULT '{}',

    -- Timestamps
    timestamp DateTime64(3),
    created_at DateTime64(3),
    updated_at DateTime64(3),
    event_ts DateTime64(3) DEFAULT now64(3),

    -- Soft delete
    is_deleted UInt8 DEFAULT 0,

    -- Indexes
    INDEX idx_trace_id trace_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_observation_id observation_id TYPE bloom_filter GRANULARITY 1,
    INDEX idx_name name TYPE bloom_filter GRANULARITY 1

) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (project_id, trace_id, timestamp, id)
PARTITION BY toYYYYMM(timestamp)
TTL timestamp + INTERVAL 365 DAY;


-- ============================================================
-- AGGREGATED VIEWS (Materialized Views)
-- For dashboard queries
-- ============================================================

-- Daily aggregates per project
CREATE MATERIALIZED VIEW IF NOT EXISTS observations_daily_mv
ENGINE = SummingMergeTree()
ORDER BY (project_id, date, model, type)
AS SELECT
    project_id,
    toDate(start_time) as date,
    model,
    type,
    count() as request_count,
    sum(input_tokens) as total_input_tokens,
    sum(output_tokens) as total_output_tokens,
    sum(total_cost) as total_cost,
    avg(if(end_time IS NOT NULL, dateDiff('millisecond', start_time, end_time), 0)) as avg_latency_ms,
    quantile(0.95)(if(end_time IS NOT NULL, dateDiff('millisecond', start_time, end_time), 0)) as p95_latency_ms
FROM observations
WHERE is_deleted = 0
GROUP BY project_id, date, model, type;


-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Function to calculate cost from token count (placeholder - configure with your pricing)
-- Example: $0.01 per 1K input tokens, $0.03 per 1K output tokens for GPT-4
-- This should be customized based on your model pricing

-- Note: ClickHouse doesn't support user-defined functions in the same way as PostgreSQL.
-- Cost calculation should be done in application code or via a models/pricing reference table.


-- ============================================================
-- SAMPLE QUERIES
-- ============================================================

-- Get total tokens and cost by model for last 7 days:
-- SELECT
--     model,
--     count() as requests,
--     sum(input_tokens) as total_input,
--     sum(output_tokens) as total_output,
--     sum(total_cost) as total_cost
-- FROM observations
-- WHERE project_id = 'your-project'
--   AND start_time > now() - INTERVAL 7 DAY
--   AND type = 'GENERATION'
--   AND is_deleted = 0
-- GROUP BY model
-- ORDER BY total_cost DESC;

-- Get latency percentiles:
-- SELECT
--     quantile(0.50)(dateDiff('millisecond', start_time, end_time)) as p50,
--     quantile(0.95)(dateDiff('millisecond', start_time, end_time)) as p95,
--     quantile(0.99)(dateDiff('millisecond', start_time, end_time)) as p99
-- FROM observations
-- WHERE project_id = 'your-project'
--   AND type = 'GENERATION'
--   AND end_time IS NOT NULL
--   AND is_deleted = 0;
