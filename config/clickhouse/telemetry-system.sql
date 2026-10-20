-- System-only projections and pricing; tenant grants remain manifest-owned.
CREATE DATABASE IF NOT EXISTS telemetry;

CREATE OR REPLACE VIEW telemetry.span_annotations SQL SECURITY INVOKER AS
SELECT *, workspace_id AS project_id FROM telemetry.annotations FINAL;

CREATE TABLE IF NOT EXISTS telemetry.event_details (
    timestamp DateTime64(9), service String, body String, severity String,
    event_id String, event_type String, event_source String, source_type String,
    job_id String, job_tag String, status String, api_key String, message String, metadata_json String
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (source_type, event_type, timestamp)
TTL toDateTime(timestamp) + INTERVAL 30 DAY;

CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry.events_mv TO telemetry.event_details AS
SELECT Timestamp AS timestamp, ServiceName AS service, Body AS body, SeverityText AS severity,
       LogAttributes['event.id'] AS event_id, LogAttributes['event.type'] AS event_type,
       LogAttributes['event.source'] AS event_source, LogAttributes['event.source_type'] AS source_type,
       LogAttributes['job.id'] AS job_id, LogAttributes['job.tag'] AS job_tag,
       LogAttributes['status'] AS status, LogAttributes['api_key'] AS api_key,
       LogAttributes['payload.message'] AS message, LogAttributes['payload.metadata'] AS metadata_json
FROM telemetry.logs WHERE length(LogAttributes['event.id']) > 0;

CREATE TABLE IF NOT EXISTS telemetry.capacity_ts (
    timestamp DateTime64(9), total_capacity Int64, used Int64, available Int64, holders Int64
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY timestamp
TTL toDateTime(timestamp) + INTERVAL 7 DAY;

CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry.capacity_ts_ingest TO telemetry.capacity_ts AS
SELECT Timestamp AS timestamp,
       JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 1, 2, 'capacity') AS total_capacity,
       JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 1, 2, 'used') AS used,
       JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 1, 2, 'available') AS available,
       JSONExtractInt(JSONExtractRaw(LogAttributes['payload.metadata'], 'capacity'), 1, 2, 'holder_count') AS holders
FROM telemetry.logs WHERE LogAttributes['job.tag'] = 'RESOURCE_EXECUTOR_UPDATED';

CREATE TABLE IF NOT EXISTS telemetry.model_prices (
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

CREATE TABLE IF NOT EXISTS telemetry.model_price_sync_runs (
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

CREATE DICTIONARY IF NOT EXISTS telemetry.model_price_dict (
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
SOURCE(CLICKHOUSE(NAME telemetry_prices QUERY '
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
        FROM (SELECT DISTINCT model, provider FROM telemetry.model_names) AS n
        CROSS JOIN (SELECT * FROM telemetry.model_prices FINAL) AS p
        WHERE match(n.model, concat(''(?i)'', p.match_pattern))
            AND (p.provider = '''' OR lower(p.provider) = lower(n.provider))
            AND (p.start_date IS NULL OR p.start_date <= now64(3))
        GROUP BY n.model, n.provider
    )
'))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(MIN 30 MAX 60);

