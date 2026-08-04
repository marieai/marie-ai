-- File: 045_asset_functions.sql
-- Description: Helper functions and triggers for asset management
-- Dependencies: 044_asset_tables.sql

-- ============================================================================
-- Trigger: Auto-update asset_latest on new materialization
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.update_asset_latest()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO {schema}.asset_latest (asset_key, latest_materialization_id, latest_version, latest_at, partition_key)
    VALUES (NEW.asset_key, NEW.id, COALESCE(NEW.asset_version, ''), NEW.created_at, NEW.partition_key)
    ON CONFLICT (asset_key) DO UPDATE SET
        latest_materialization_id = EXCLUDED.latest_materialization_id,
        latest_version = EXCLUDED.latest_version,
        latest_at = EXCLUDED.latest_at,
        partition_key = EXCLUDED.partition_key
    WHERE EXCLUDED.latest_at > {schema}.asset_latest.latest_at;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_asset_latest ON {schema}.asset_materialization;
CREATE TRIGGER trigger_update_asset_latest
    AFTER INSERT ON {schema}.asset_materialization
    FOR EACH ROW
    EXECUTE FUNCTION {schema}.update_asset_latest();

-- ============================================================================
-- View: Asset Freshness - computed freshness status
-- ============================================================================

CREATE OR REPLACE VIEW {schema}.asset_freshness AS
SELECT
    ar.asset_key,
    ar.kind,
    al.latest_version,
    al.latest_at,
    EXTRACT(EPOCH FROM (NOW() - al.latest_at))::INTEGER as age_seconds,
    CASE
        WHEN al.latest_at IS NULL THEN 'never_materialized'
        WHEN al.latest_at >= NOW() - INTERVAL '1 hour' THEN 'fresh'
        WHEN al.latest_at >= NOW() - INTERVAL '24 hours' THEN 'stale'
        ELSE 'very_stale'
    END as freshness_status
FROM {schema}.asset_registry ar
LEFT JOIN {schema}.asset_latest al ON ar.asset_key = al.asset_key;

COMMENT ON VIEW {schema}.asset_freshness IS 'Computed view of asset freshness status';

-- ============================================================================
-- Function: Get upstream assets for a DAG node
-- ============================================================================

CREATE OR REPLACE FUNCTION {schema}.get_upstream_assets_for_node(
    p_dag_id UUID,
    p_node_task_id VARCHAR(255)
)
RETURNS TABLE (
    asset_key VARCHAR(255),
    latest_version VARCHAR(100),
    partition_key VARCHAR(255)
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT
        dam.asset_key,
        al.latest_version,
        al.partition_key
    FROM {schema}.dag_asset_map dam
    LEFT JOIN {schema}.asset_latest al ON dam.asset_key = al.asset_key
    WHERE dam.dag_id = p_dag_id
      AND dam.node_task_id = ANY(
          SELECT unnest(dm.upstream_nodes)
          FROM {schema}.dag_asset_map dm
          WHERE dm.dag_id = p_dag_id AND dm.node_task_id = p_node_task_id
      );
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION {schema}.get_upstream_assets_for_node(UUID, VARCHAR) IS 'Get upstream assets that a node depends on';
