"""Core models and types for asset tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Type alias for asset keys
AssetKey = Tuple[str, str]  # (namespace, path)


class AssetKind(str, Enum):
    """Standard asset types in Marie-AI (data format kinds)."""

    TEXT = "text"
    JSON = "json"
    BBOX = "bbox"
    CLASSIFICATION = "classification"
    VECTOR = "vector"
    IMAGE = "image"
    TABLE = "table"
    METADATA = "metadata"
    BLOB = "blob"


class AssetTechnologyKind(str, Enum):
    """
    Technology/tool kinds for assets.

    These can be combined with AssetKind to provide both data format
    and technology context. Use the `kinds` field in AssetSpec to
    specify multiple kinds.
    """

    PYTHON = "python"
    SQL = "sql"
    DBT = "dbt"
    SNOWFLAKE = "snowflake"
    POSTGRES = "postgres"
    S3 = "s3"
    API = "api"
    ML_MODEL = "ml_model"
    LLM = "llm"
    PANDAS = "pandas"
    NUMPY = "numpy"
    TORCH = "torch"
    ONNX = "onnx"
    CRAFT_OCR = "craft_ocr"
    TESSERACT = "tesseract"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class AssetSpec:
    """
    Specification for a single asset produced by a node.

    Used during DAG submission to pre-register expected assets.

    Supports multi-kind tagging via the `kinds` field:
        - Use `kind` for the primary data format (TEXT, JSON, BBOX, etc.)
        - Use `kinds` to add technology/tool tags (python, llm, torch, etc.)

    Example:
        AssetSpec(
            key="ocr/text",
            kind=AssetKind.TEXT,
            kinds=["craft_ocr", "torch", "python"],  # Multiple technology kinds
            metadata={"model": "craft-v2", "language": "en"}
        )
    """

    key: str  # e.g., "ocr/text"
    kind: AssetKind
    is_primary: bool = True  # Is this the main output?
    is_required: bool = True  # Is this critical for data quality?
    description: Optional[str] = None
    kinds: List[str] = field(default_factory=list)  # Additional technology kinds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "key": self.key,
            "kind": self.kind.value if isinstance(self.kind, AssetKind) else self.kind,
            "is_primary": self.is_primary,
            "is_required": self.is_required,
            "description": self.description,
            "kinds": self.kinds,
            "metadata": self.metadata,
        }

    def add_kind(self, kind: str) -> "AssetSpec":
        """Add a technology/tool kind tag (chainable)."""
        if kind not in self.kinds:
            self.kinds.append(kind)
        return self

    def has_kind(self, kind: str) -> bool:
        """Check if asset has a specific kind tag."""
        return kind in self.kinds or (
            self.kind.value == kind
            if isinstance(self.kind, AssetKind)
            else self.kind == kind
        )

    def get_all_kinds(self) -> List[str]:
        """Get all kinds including primary kind and additional kinds."""
        primary = self.kind.value if isinstance(self.kind, AssetKind) else self.kind
        return [primary] + self.kinds


@dataclass
class NodeAssetSpec:
    """
    Complete asset specification for a DAG node.
    Defines all assets that a node can/should produce.
    """

    node_task_id: str
    assets: List[AssetSpec]

    @property
    def primary_asset(self) -> Optional[AssetSpec]:
        """Get the primary asset."""
        for asset in self.assets:
            if asset.is_primary:
                return asset
        return self.assets[0] if self.assets else None

    @property
    def required_assets(self) -> List[AssetSpec]:
        """Get all required assets."""
        return [a for a in self.assets if a.is_required]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "node_task_id": self.node_task_id,
            "assets": [a.to_dict() for a in self.assets],
        }


@dataclass
class AssetInfo:
    """Asset registry entry."""

    id: int
    asset_key: str
    namespace: str
    kind: str
    description: Optional[str]
    tags: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class AssetVersion:
    """Asset version information."""

    asset_key: str
    version: str  # Content hash
    created_at: datetime


@dataclass
class MaterializationInfo:
    """Asset materialization event."""

    id: int
    storage_event_id: Optional[int]
    asset_key: str
    asset_version: Optional[str]
    job_id: str
    dag_id: Optional[str]
    node_task_id: Optional[str]
    partition_key: Optional[str]
    size_bytes: Optional[int]
    checksum: Optional[str]
    uri: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class LineageInfo:
    """Asset lineage edge."""

    upstream_asset_key: str
    upstream_version: Optional[str]
    upstream_partition_key: Optional[str]


@dataclass
class AssetLatestInfo:
    """Latest version information for an asset."""

    asset_key: str
    version: str
    latest_at: datetime
    partition_key: Optional[str] = None


@dataclass
class NodeMaterializationStatus:
    """Materialization status for a DAG node."""

    dag_id: str
    dag_name: str
    node_task_id: str
    expected_assets: int
    materialized_assets: int
    required_assets: int
    materialized_required: int
    all_required_materialized: bool
    missing_required_assets: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if all required assets are materialized."""
        return self.all_required_materialized

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.required_assets == 0:
            return 100.0
        return (self.materialized_required / self.required_assets) * 100.0


@dataclass
class UpstreamAssetInfo:
    """Information about an upstream asset."""

    asset_key: str
    version: Optional[str]
    partition_key: Optional[str]


@dataclass
class AssetLineage:
    """Complete lineage information for an asset."""

    asset_key: str
    upstream: List[UpstreamAssetInfo] = field(default_factory=list)
    downstream: List[str] = field(
        default_factory=list
    )  # Asset keys that depend on this


# Predefined asset specs for common node types
OCR_NODE_ASSETS = NodeAssetSpec(
    node_task_id="ocr",
    assets=[
        AssetSpec(
            key="ocr/text",
            kind=AssetKind.TEXT,
            is_primary=True,
            description="Extracted text from OCR",
        ),
        AssetSpec(
            key="ocr/bboxes",
            kind=AssetKind.BBOX,
            is_primary=False,
            description="Bounding boxes for text regions",
        ),
        AssetSpec(
            key="ocr/confidence",
            kind=AssetKind.METADATA,
            is_required=False,
            description="OCR confidence scores",
        ),
    ],
)

EXTRACT_NODE_ASSETS = NodeAssetSpec(
    node_task_id="extract",
    assets=[
        AssetSpec(
            key="extract/claims",
            kind=AssetKind.JSON,
            is_primary=True,
            description="Extracted claims data",
        ),
        AssetSpec(
            key="extract/headers",
            kind=AssetKind.JSON,
            is_primary=False,
            description="Extracted header information",
        ),
        AssetSpec(
            key="extract/service_lines",
            kind=AssetKind.TABLE,
            is_primary=False,
            description="Extracted service line items",
        ),
        AssetSpec(
            key="extract/metadata",
            kind=AssetKind.METADATA,
            is_required=False,
            description="Extraction metadata and statistics",
        ),
    ],
)

CLASSIFY_NODE_ASSETS = NodeAssetSpec(
    node_task_id="classify",
    assets=[
        AssetSpec(
            key="classify/document_type",
            kind=AssetKind.CLASSIFICATION,
            is_primary=True,
            description="Document classification result",
        ),
        AssetSpec(
            key="classify/confidence_scores",
            kind=AssetKind.METADATA,
            is_required=False,
            description="Classification confidence scores",
        ),
    ],
)

LOCATE_NODE_ASSETS = NodeAssetSpec(
    node_task_id="locate",
    assets=[
        AssetSpec(
            key="locate/tables",
            kind=AssetKind.BBOX,
            is_primary=True,
            description="Located table regions",
        ),
        AssetSpec(
            key="locate/forms",
            kind=AssetKind.BBOX,
            is_primary=False,
            description="Located form regions",
        ),
    ],
)

# Registry of predefined asset specs by node type pattern
PREDEFINED_ASSET_SPECS = {
    "ocr": OCR_NODE_ASSETS,
    "extract": EXTRACT_NODE_ASSETS,
    "classify": CLASSIFY_NODE_ASSETS,
    "locate": LOCATE_NODE_ASSETS,
}
