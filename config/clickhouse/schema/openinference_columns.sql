-- Materialized columns for fast OI queries without Map access
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

-- Indexes for common query patterns
ALTER TABLE otel.otel_traces ADD INDEX IF NOT EXISTS
    idx_oi_span_kind oi_span_kind TYPE set(10) GRANULARITY 4;
ALTER TABLE otel.otel_traces ADD INDEX IF NOT EXISTS
    idx_marie_project_id marie_project_id TYPE bloom_filter(0.001) GRANULARITY 1;
ALTER TABLE otel.otel_traces ADD INDEX IF NOT EXISTS
    idx_llm_model_name llm_model_name TYPE set(50) GRANULARITY 4;
ALTER TABLE otel.otel_traces ADD INDEX IF NOT EXISTS
    idx_oi_session_id oi_session_id TYPE bloom_filter(0.001) GRANULARITY 1;
