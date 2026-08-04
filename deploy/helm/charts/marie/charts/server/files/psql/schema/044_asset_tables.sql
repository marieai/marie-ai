-- File: 044_asset_tables.sql
-- Description: Asset tracking tables for materialization, lineage, and quality checks
-- Dependencies: 001_schema.sql, 007_dag.sql, 005_job.sql
--
-- These tables track asset materializations produced during job execution,
-- enabling asset lineage visualization and freshness monitoring.
--
-- Previously managed by marie-studio in marie_studio schema, now owned by marie-ai.

-- ============================================================================
-- Asset Registry - catalog of all assets
-- ============================================================================

CREATE TABLE IF NOT EXISTS {schema}.asset_registry (
    id BIGSERIAL PRIMARY KEY,
    asset_key VARCHAR(255) NOT NULL UNIQUE,
    namespace VARCHAR(100) NOT NULL DEFAULT 'marie-ai',
    kind VARCHAR(50) NOT NULL,
    description TEXT,
    tags JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asset_registry_kind ON {schema}.asset_registry(kind);
CREATE INDEX IF NOT EXISTS idx_asset_registry_namespace ON {schema}.asset_registry(namespace);

COMMENT ON TABLE {schema}.asset_registry IS 'Catalog of all tracked assets with metadata';

-- ============================================================================
-- DAG Asset Map - pre-registered assets for each DAG node
-- ============================================================================

CREATE TABLE IF NOT EXISTS {schema}.dag_asset_map (
    id BIGSERIAL PRIMARY KEY,
    dag_id UUID NOT NULL REFERENCES {schema}.dag(id) ON DELETE CASCADE,
    dag_name VARCHAR(255) NOT NULL,
    node_task_id VARCHAR(255) NOT NULL,
    asset_key VARCHAR(255) NOT NULL,
    kind VARCHAR(50) NOT NULL,
    job_level INTEGER NOT NULL,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    is_required BOOLEAN NOT NULL DEFAULT TRUE,
    upstream_nodes TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(dag_id, node_task_id, asset_key)
);

CREATE INDEX IF NOT EXISTS idx_dag_asset_map_dag_node ON {schema}.dag_asset_map(dag_id, node_task_id);
CREATE INDEX IF NOT EXISTS idx_dag_asset_map_asset_key ON {schema}.dag_asset_map(asset_key);
CREATE INDEX IF NOT EXISTS idx_dag_asset_map_dag_id ON {schema}.dag_asset_map(dag_id);

COMMENT ON TABLE {schema}.dag_asset_map IS 'Pre-registered assets expected from each DAG node';

-- ============================================================================
-- Asset Materialization - records when assets are produced
-- ============================================================================

CREATE TABLE IF NOT EXISTS {schema}.asset_materialization (
    id BIGSERIAL PRIMARY KEY,
    storage_event_id BIGINT,
    asset_key VARCHAR(255) NOT NULL,
    asset_version VARCHAR(100),
    job_id UUID NOT NULL,
    dag_id UUID,
    node_task_id VARCHAR(255),
    partition_key VARCHAR(255),
    size_bytes BIGINT,
    checksum VARCHAR(128),
    uri TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(storage_event_id, asset_key)
);

CREATE INDEX IF NOT EXISTS idx_asset_mat_key_created ON {schema}.asset_materialization(asset_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_asset_mat_job_dag ON {schema}.asset_materialization(job_id, dag_id);
CREATE INDEX IF NOT EXISTS idx_asset_mat_node ON {schema}.asset_materialization(node_task_id);
CREATE INDEX IF NOT EXISTS idx_asset_mat_key_partition ON {schema}.asset_materialization(asset_key, partition_key);
CREATE INDEX IF NOT EXISTS idx_asset_mat_key_version ON {schema}.asset_materialization(asset_key, asset_version);

COMMENT ON TABLE {schema}.asset_materialization IS 'Records of when assets were produced by jobs';

-- ============================================================================
-- Asset Lineage - tracks upstream dependencies
-- ============================================================================

CREATE TABLE IF NOT EXISTS {schema}.asset_lineage (
    id BIGSERIAL PRIMARY KEY,
    materialization_id BIGINT NOT NULL REFERENCES {schema}.asset_materialization(id) ON DELETE CASCADE,
    upstream_asset_key VARCHAR(255) NOT NULL,
    upstream_version VARCHAR(100),
    upstream_partition_key VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asset_lineage_mat ON {schema}.asset_lineage(materialization_id);
CREATE INDEX IF NOT EXISTS idx_asset_lineage_upstream ON {schema}.asset_lineage(upstream_asset_key);

COMMENT ON TABLE {schema}.asset_lineage IS 'Tracks upstream asset dependencies for lineage visualization';

-- ============================================================================
-- Asset Latest - fast lookup for latest versions
-- ============================================================================

CREATE TABLE IF NOT EXISTS {schema}.asset_latest (
    asset_key VARCHAR(255) PRIMARY KEY,
    latest_materialization_id BIGINT NOT NULL,
    latest_version VARCHAR(100) NOT NULL,
    latest_at TIMESTAMPTZ NOT NULL,
    partition_key VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_asset_latest_at ON {schema}.asset_latest(latest_at DESC);

COMMENT ON TABLE {schema}.asset_latest IS 'Fast lookup table for latest asset versions';

-- ============================================================================
-- Asset Check Result - quality checks/metrics
-- ============================================================================

CREATE TABLE IF NOT EXISTS {schema}.asset_check_result (
    id BIGSERIAL PRIMARY KEY,
    materialization_id BIGINT NOT NULL REFERENCES {schema}.asset_materialization(id) ON DELETE CASCADE,
    asset_key VARCHAR(255) NOT NULL,
    partition_key VARCHAR(255),
    check_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asset_check_key_name_status ON {schema}.asset_check_result(asset_key, check_name, status);
CREATE INDEX IF NOT EXISTS idx_asset_check_mat ON {schema}.asset_check_result(materialization_id);

COMMENT ON TABLE {schema}.asset_check_result IS 'Quality check results for asset materializations';
