-- ===================================================================
-- Marie-AI Asset Tracking Schema
-- Multi-Asset Support for DAG Nodes
--
-- PURPOSE: Track what was produced (lineage, versioning, caching)
-- NOT FOR: Controlling job execution (that's the scheduler's job)
-- ===================================================================

-- Set search path
SET search_path = marie_scheduler, public, pg_catalog;

-- ===================================================================
-- 1) Asset Registry (catalog of all assets)
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.asset_registry (
  id                 BIGSERIAL PRIMARY KEY,
  asset_key          TEXT        NOT NULL UNIQUE,
  namespace          TEXT        NOT NULL DEFAULT 'marie-ai',
  kind               TEXT        NOT NULL,  -- 'text', 'json', 'bbox', 'classification', etc.
  description        TEXT        NULL,
  tags               JSONB       NULL DEFAULT '{}'::jsonb,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_asset_registry_kind ON marie_scheduler.asset_registry(kind);
CREATE INDEX IF NOT EXISTS idx_asset_registry_namespace ON marie_scheduler.asset_registry(namespace);

COMMENT ON TABLE marie_scheduler.asset_registry IS 'Catalog of all assets in the system';
COMMENT ON COLUMN marie_scheduler.asset_registry.asset_key IS 'Unique asset identifier (e.g., ocr/text, extract/claims)';
COMMENT ON COLUMN marie_scheduler.asset_registry.kind IS 'Asset type (text, json, bbox, classification, vector, etc.)';

-- ===================================================================
-- 2) DAG Asset Map - REMOVED
--    No longer pre-defining assets per node
--    Assets are tracked dynamically as they are materialized
-- ===================================================================

-- ===================================================================
-- 3) Asset Materialization Events
--    Records when assets are actually produced
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.asset_materialization (
  id                 BIGSERIAL PRIMARY KEY,
  storage_event_id   BIGINT      NULL,  -- Links to storage table event_id (if using PostgreSQLStorage)
  asset_key          TEXT        NOT NULL,
  asset_version      TEXT        NULL,
  job_id             UUID        NOT NULL,
  dag_id             UUID        NULL,
  node_task_id       TEXT        NULL,
  partition_key      TEXT        NULL,
  size_bytes         BIGINT      NULL,
  checksum           TEXT        NULL,
  uri                TEXT        NULL,  -- Location of the asset (s3://, file://, etc.)
  metadata           JSONB       NULL DEFAULT '{}'::jsonb,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Add unique constraint only if storage_event_id is not null
CREATE UNIQUE INDEX IF NOT EXISTS idx_mat_storage_event_asset
  ON marie_scheduler.asset_materialization(storage_event_id, asset_key)
  WHERE storage_event_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mat_asset_key ON marie_scheduler.asset_materialization(asset_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mat_job_dag ON marie_scheduler.asset_materialization(job_id, dag_id);
CREATE INDEX IF NOT EXISTS idx_mat_node_task ON marie_scheduler.asset_materialization(node_task_id);
CREATE INDEX IF NOT EXISTS idx_mat_partition ON marie_scheduler.asset_materialization(asset_key, partition_key);
CREATE INDEX IF NOT EXISTS idx_mat_version ON marie_scheduler.asset_materialization(asset_key, asset_version);

COMMENT ON TABLE marie_scheduler.asset_materialization IS 'Historical record of asset materializations';
COMMENT ON COLUMN marie_scheduler.asset_materialization.storage_event_id IS 'Links to storage table event_id (optional)';
COMMENT ON COLUMN marie_scheduler.asset_materialization.asset_version IS 'Content-addressed version (e.g., v:sha256:...)';

-- ===================================================================
-- 4) Asset Lineage (tracks dependencies between assets)
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.asset_lineage (
  id                        BIGSERIAL PRIMARY KEY,
  materialization_id        BIGINT      NOT NULL REFERENCES marie_scheduler.asset_materialization(id) ON DELETE CASCADE,
  upstream_asset_key        TEXT        NOT NULL,
  upstream_version          TEXT        NULL,
  upstream_partition_key    TEXT        NULL,
  created_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lineage_mat ON marie_scheduler.asset_lineage(materialization_id);
CREATE INDEX IF NOT EXISTS idx_lineage_upstream ON marie_scheduler.asset_lineage(upstream_asset_key);

COMMENT ON TABLE marie_scheduler.asset_lineage IS 'Tracks upstream dependencies between assets (for lineage visualization)';

-- ===================================================================
-- 5) Asset Latest Pointers (for fast "latest version" queries)
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.asset_latest (
  asset_key                  TEXT PRIMARY KEY,
  latest_materialization_id  BIGINT NOT NULL REFERENCES marie_scheduler.asset_materialization(id) ON DELETE RESTRICT,
  latest_version             TEXT NOT NULL,
  latest_at                  TIMESTAMPTZ NOT NULL,
  partition_key              TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_asset_latest_at ON marie_scheduler.asset_latest(latest_at DESC);

COMMENT ON TABLE marie_scheduler.asset_latest IS 'Fast lookup for latest version of each asset (for caching decisions)';

-- ===================================================================
-- 6) Asset Checks (quality checks/metrics)
-- ===================================================================
CREATE TABLE IF NOT EXISTS marie_scheduler.asset_check_result (
  id                 BIGSERIAL PRIMARY KEY,
  materialization_id BIGINT      NOT NULL REFERENCES marie_scheduler.asset_materialization(id) ON DELETE CASCADE,
  asset_key          TEXT        NOT NULL,
  partition_key      TEXT        NULL,
  check_name         TEXT        NOT NULL,   -- 'row_count>0', 'schema_match', 'freshness'
  status             TEXT        NOT NULL,   -- 'passed', 'failed', 'skipped'
  details            JSONB       NULL,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_check_asset ON marie_scheduler.asset_check_result(asset_key, check_name, status);
CREATE INDEX IF NOT EXISTS idx_check_mat ON marie_scheduler.asset_check_result(materialization_id);

COMMENT ON TABLE marie_scheduler.asset_check_result IS 'Quality check results for assets (informational)';

-- ===================================================================
-- 7) Helper Views (for observability and analytics)
-- ===================================================================

-- View: Node Materialization Status
-- Simplified to show actual assets produced by each node
CREATE OR REPLACE VIEW marie_scheduler.node_materialization_status AS
SELECT
  am.dag_id,
  d.name as dag_name,
  am.node_task_id,
  COUNT(DISTINCT am.asset_key) as materialized_assets,
  ARRAY_AGG(DISTINCT am.asset_key ORDER BY am.asset_key) as asset_keys,
  MAX(am.created_at) as last_materialized_at
FROM marie_scheduler.asset_materialization am
LEFT JOIN marie_scheduler.dag d ON d.id = am.dag_id
WHERE am.dag_id IS NOT NULL
GROUP BY am.dag_id, d.name, am.node_task_id;

COMMENT ON VIEW marie_scheduler.node_materialization_status IS 'Shows actual assets materialized by each DAG node (tracks what was produced, not what was expected)';

-- View: Asset Lineage Graph
CREATE OR REPLACE VIEW marie_scheduler.asset_lineage_graph AS
SELECT
  am.asset_key,
  am.asset_version,
  am.created_at as materialized_at,
  am.job_id,
  am.dag_id,
  am.node_task_id,
  al.upstream_asset_key,
  al.upstream_version,
  al.upstream_partition_key
FROM marie_scheduler.asset_materialization am
LEFT JOIN marie_scheduler.asset_lineage al ON al.materialization_id = am.id;

COMMENT ON VIEW marie_scheduler.asset_lineage_graph IS 'Complete asset lineage graph for visualization';

-- View: Asset Freshness
CREATE OR REPLACE VIEW marie_scheduler.asset_freshness AS
SELECT
  ar.asset_key,
  ar.kind,
  al.latest_version,
  al.latest_at,
  EXTRACT(EPOCH FROM (now() - al.latest_at)) as age_seconds,
  CASE
    WHEN al.latest_at IS NULL THEN 'never_materialized'
    WHEN al.latest_at > now() - interval '1 hour' THEN 'fresh'
    WHEN al.latest_at > now() - interval '24 hours' THEN 'stale'
    ELSE 'very_stale'
  END as freshness_status
FROM marie_scheduler.asset_registry ar
LEFT JOIN marie_scheduler.asset_latest al ON al.asset_key = ar.asset_key;

COMMENT ON VIEW marie_scheduler.asset_freshness IS 'Shows freshness of each asset';

-- ===================================================================
-- 8) Helper Functions (for queries and analytics only)
-- ===================================================================

-- Function: Get upstream assets for a DAG node (from actual lineage)
CREATE OR REPLACE FUNCTION marie_scheduler.get_upstream_assets_for_node(
  p_dag_id UUID,
  p_node_task_id TEXT
)
RETURNS TABLE (
  asset_key TEXT,
  latest_version TEXT,
  partition_key TEXT
) AS $$
BEGIN
  -- Query actual lineage from asset_lineage table
  -- This returns the upstream assets that were actually used to produce
  -- the assets materialized by this node
  RETURN QUERY
  SELECT DISTINCT
    al.upstream_asset_key as asset_key,
    latest.latest_version,
    latest.partition_key
  FROM marie_scheduler.asset_materialization am
  INNER JOIN marie_scheduler.asset_lineage al ON al.materialization_id = am.id
  LEFT JOIN marie_scheduler.asset_latest latest ON latest.asset_key = al.upstream_asset_key
  WHERE am.dag_id = p_dag_id
    AND am.node_task_id = p_node_task_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION marie_scheduler.get_upstream_assets_for_node IS 'Get upstream assets from actual lineage (tracks what was actually consumed)';

-- Function: Get asset materialization history
CREATE OR REPLACE FUNCTION marie_scheduler.get_asset_history(
  p_asset_key TEXT,
  p_limit INT DEFAULT 10
)
RETURNS TABLE (
  materialization_id BIGINT,
  asset_version TEXT,
  job_id UUID,
  dag_id UUID,
  node_task_id TEXT,
  partition_key TEXT,
  size_bytes BIGINT,
  checksum TEXT,
  uri TEXT,
  created_at TIMESTAMPTZ
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    am.id,
    am.asset_version,
    am.job_id,
    am.dag_id,
    am.node_task_id,
    am.partition_key,
    am.size_bytes,
    am.checksum,
    am.uri,
    am.created_at
  FROM marie_scheduler.asset_materialization am
  WHERE am.asset_key = p_asset_key
  ORDER BY am.created_at DESC
  LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION marie_scheduler.get_asset_history IS 'Get materialization history for an asset';

-- ===================================================================
-- 9) Grant permissions (adjust as needed)
-- ===================================================================

-- Grant access to application role (uncomment and adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA marie_scheduler TO marie_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA marie_scheduler TO marie_app;
