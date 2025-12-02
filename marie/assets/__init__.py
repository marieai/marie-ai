"""
Marie-AI Asset Tracking System

Provides DAG-aware asset tracking with:
- Multi-asset support per node
- Automatic lineage tracking
- Content-addressed versioning
- Minimal configuration
- Asset decorators (@asset, @multi_asset, etc.)

For observability and caching - NOT for execution control.
"""

from .context import (
    AssetExecutionContext,
    AssetMaterializationContext,
    build_asset_context,
)
from .dag_mapper import DAGAssetMapper
from .decorators import (
    asset,
    get_asset_spec,
    get_asset_specs,
    graph_asset,
    is_asset,
    is_graph_asset,
    is_multi_asset,
    is_observable_asset,
    multi_asset,
    observable_source_asset,
)
from .models import (
    AssetInfo,
    AssetKey,
    AssetKind,
    AssetLatestInfo,
    AssetLineage,
    AssetSpec,
    AssetTechnologyKind,
    LineageInfo,
    MaterializationInfo,
    NodeAssetSpec,
    NodeMaterializationStatus,
    UpstreamAssetInfo,
)
from .repository import AssetRepository
from .tracker import AssetTracker

__all__ = [
    # Core classes
    "AssetTracker",
    "DAGAssetMapper",
    "AssetRepository",
    # Context
    "AssetExecutionContext",
    "AssetMaterializationContext",
    "build_asset_context",
    # Decorators
    "asset",
    "multi_asset",
    "graph_asset",
    "observable_source_asset",
    "get_asset_spec",
    "get_asset_specs",
    "is_asset",
    "is_multi_asset",
    "is_graph_asset",
    "is_observable_asset",
    # Models
    "AssetSpec",
    "NodeAssetSpec",
    "AssetInfo",
    "AssetLatestInfo",
    "MaterializationInfo",
    "LineageInfo",
    "NodeMaterializationStatus",
    "UpstreamAssetInfo",
    "AssetLineage",
    # Types/Enums
    "AssetKey",
    "AssetKind",
    "AssetTechnologyKind",
]
