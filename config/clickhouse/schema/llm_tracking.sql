-- ############### LLM Tracking ClickHouse Schema ###############
--
-- Tables for storing LLM observability data:
-- - traces: Root containers for LLM interactions
-- - observations: Individual observations (spans, generations, events)
-- - scores: Evaluation scores for traces or observations
--
-- Usage:
--   docker exec -i marie-clickhouse clickhouse-client --database marie < config/clickhouse/schema/llm_tracking.sql

-- Traces table - Root container for LLM interactions
CREATE TABLE IF NOT EXISTS traces (
    id UUID,
    name String,
    project_id String,
    user_id Nullable(String),
    session_id Nullable(String),
    input Nullable(String),
    output Nullable(String),
    metadata String DEFAULT '{}',
    tags Array(String) DEFAULT [],
    timestamp DateTime64(3),
    created_at DateTime64(3),
    updated_at DateTime64(3),
    release Nullable(String),
    version Nullable(String),
    is_deleted UInt8 DEFAULT 0
) ENGINE = MergeTree()
ORDER BY (project_id, timestamp, id)
SETTINGS index_granularity = 8192;

-- Observations table - Spans, generations, and events within a trace
CREATE TABLE IF NOT EXISTS observations (
    id UUID,
    trace_id UUID,
    project_id String,
    parent_observation_id Nullable(UUID),
    type String,
    name String,
    start_time DateTime64(3),
    end_time Nullable(DateTime64(3)),
    completion_start_time Nullable(DateTime64(3)),
    model Nullable(String),
    model_parameters Nullable(String),
    input Nullable(String),
    output Nullable(String),
    metadata String DEFAULT '{}',
    level String DEFAULT 'DEFAULT',
    status_message Nullable(String),
    version Nullable(String),
    input_tokens Nullable(Int64),
    output_tokens Nullable(Int64),
    total_tokens Nullable(Int64),
    usage_details String DEFAULT '{}',
    input_cost Nullable(Float64),
    output_cost Nullable(Float64),
    total_cost Nullable(Float64),
    cost_details String DEFAULT '{}',
    created_at DateTime64(3),
    updated_at DateTime64(3),
    is_deleted UInt8 DEFAULT 0
) ENGINE = MergeTree()
ORDER BY (project_id, trace_id, start_time, id)
SETTINGS index_granularity = 8192;

-- Scores table - Evaluation scores for traces or observations
CREATE TABLE IF NOT EXISTS scores (
    id UUID,
    trace_id UUID,
    observation_id Nullable(UUID),
    project_id String,
    name String,
    value Float64,
    data_type String DEFAULT 'NUMERIC',
    source String DEFAULT 'API',
    comment Nullable(String),
    metadata String DEFAULT '{}',
    timestamp DateTime64(3),
    created_at DateTime64(3),
    updated_at DateTime64(3),
    is_deleted UInt8 DEFAULT 0
) ENGINE = MergeTree()
ORDER BY (project_id, trace_id, timestamp, id)
SETTINGS index_granularity = 8192;
