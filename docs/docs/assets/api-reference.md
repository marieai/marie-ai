---
sidebar_position: 5
---

# API Reference

Complete API documentation for Marie-AI's asset tracking system.

## Core Classes

### AssetTracker

Records asset materializations and manages lineage.

```python
from marie.assets import AssetTracker

tracker = AssetTracker(
    storage_handler=storage_handler,
    storage_conf=storage_conf
)
```

#### Methods

##### `record_materializations()`

```python
def record_materializations(
    self,
    storage_event_id: Optional[int],
    assets: List[Dict[str, Any]],
    job_id: str,
    dag_id: Optional[str] = None,
    node_task_id: Optional[str] = None,
    partition_key: Optional[str] = None,
    upstream_assets: Optional[List[Tuple[str, str, str]]] = None,
)
```

Records multiple asset materializations from a single storage event.

**Parameters:**
- `storage_event_id` - ID from storage table (optional)
- `assets` - List of asset dicts with keys:
  - `asset_key` (str): Asset identifier
  - `version` (str): Content-addressed version
  - `kind` (str): Asset type (text, json, bbox, etc.)
  - `size_bytes` (int, optional): Size in bytes
  - `checksum` (str, optional): Content checksum
  - `uri` (str, optional): Storage location
  - `metadata` (dict, optional): Custom metadata
- `job_id` - Job ID
- `dag_id` - DAG ID (optional)
- `node_task_id` - Node task ID (optional)
- `partition_key` - Partition key (optional)
- `upstream_assets` - List of (asset_key, version, partition_key) tuples

**Returns:**
- `Future` - Async executor future returning list of (materialization_id, asset_key) tuples

**Example:**

```python
assets = [
    {
        "asset_key": "ocr/text",
        "version": "v:sha256:abc123...",
        "kind": "text",
        "size_bytes": 1024,
        "checksum": "sha256:...",
        "uri": "s3://bucket/ocr/job1/text",
        "metadata": {"language": "en"}
    }
]

upstream = [
    ("source/image", "v:sha256:def456...", None)
]

result = tracker.record_materializations(
    storage_event_id=12345,
    assets=assets,
    job_id="job-123",
    dag_id="dag-456",
    node_task_id="ocr",
    upstream_assets=upstream
)

await result
```

##### `compute_asset_version()` (static)

```python
@staticmethod
def compute_asset_version(
    payload_bytes: Optional[bytes],
    code_fingerprint: str,
    prompt_fingerprint: Optional[str] = None,
    upstream_versions: Optional[List[str]] = None,
) -> str
```

Computes deterministic content-addressed version for an asset.

**Parameters:**
- `payload_bytes` - Asset content bytes
- `code_fingerprint` - Code version (e.g., "git:abcd1234")
- `prompt_fingerprint` - Model/prompt version (e.g., "qwen3-vl:v7")
- `upstream_versions` - List of upstream asset versions

**Returns:**
- Version string like `"v:sha256:..."`

**Example:**

```python
version = AssetTracker.compute_asset_version(
    payload_bytes=text.encode('utf-8'),
    code_fingerprint="git:abcd1234",
    prompt_fingerprint="craft-ocr:v2",
    upstream_versions=["v:sha256:upstream1..."]
)
# Returns: "v:sha256:abc123..."
```

##### `cleanup()`

```python
def cleanup(self)
```

Cleanup resources (thread pool executor).

---

### AssetRepository

Query layer for asset data.

```python
from marie.assets import AssetRepository

repo = AssetRepository(
    config=db_config,
    max_workers=2
)
```

#### Methods

##### `get_asset_info()`

```python
async def get_asset_info(
    self,
    asset_key: str
) -> Optional[AssetInfo]
```

Get asset registry information.

**Parameters:**
- `asset_key` - Asset key

**Returns:**
- `AssetInfo` if found, `None` otherwise

**Example:**

```python
info = await repo.get_asset_info("ocr/text")
if info:
    print(f"Kind: {info.kind}")
    print(f"Created: {info.created_at}")
```

##### `get_latest_version()`

```python
async def get_latest_version(
    self,
    asset_key: str
) -> Optional[AssetLatestInfo]
```

Get latest version of an asset.

**Parameters:**
- `asset_key` - Asset key

**Returns:**
- `AssetLatestInfo` or `None`

**Example:**

```python
latest = await repo.get_latest_version("ocr/text")
if latest:
    print(f"Version: {latest.version}")
    print(f"Updated: {latest.latest_at}")
```

##### `get_materialization_history()`

```python
async def get_materialization_history(
    self,
    asset_key: str,
    limit: int = 10
) -> List[MaterializationInfo]
```

Get materialization history for an asset.

**Parameters:**
- `asset_key` - Asset key
- `limit` - Maximum number of records (default: 10)

**Returns:**
- List of `MaterializationInfo`

**Example:**

```python
history = await repo.get_materialization_history("ocr/text", limit=20)
for mat in history:
    print(f"{mat.created_at}: v{mat.asset_version}")
```

##### `get_lineage()`

```python
async def get_lineage(
    self,
    asset_key: str
) -> AssetLineage
```

Get lineage for an asset.

**Parameters:**
- `asset_key` - Asset key

**Returns:**
- `AssetLineage` with upstream dependencies

**Example:**

```python
lineage = await repo.get_lineage("extract/claims")
print(f"Upstream assets:")
for up in lineage.upstream:
    print(f"  - {up.asset_key} (v{up.version})")
```

##### `get_node_status()`

```python
async def get_node_status(
    self,
    dag_id: str,
    node_task_id: str
) -> Optional[NodeMaterializationStatus]
```

Get materialization status for a DAG node.

**Parameters:**
- `dag_id` - DAG ID
- `node_task_id` - Node task ID

**Returns:**
- `NodeMaterializationStatus` or `None`

**Example:**

```python
status = await repo.get_node_status(
    dag_id="550e8400-e29b-41d4-a716-446655440000",
    node_task_id="ocr"
)

if status:
    print(f"Complete: {status.is_complete}")
    print(f"Progress: {status.completion_percentage:.1f}%")
    print(f"Missing: {status.missing_required_assets}")
```

##### `get_dag_status()`

```python
async def get_dag_status(
    self,
    dag_id: str
) -> List[NodeMaterializationStatus]
```

Get materialization status for all nodes in a DAG.

**Parameters:**
- `dag_id` - DAG ID

**Returns:**
- List of `NodeMaterializationStatus` for each node

**Example:**

```python
dag_status = await repo.get_dag_status(
    dag_id="550e8400-e29b-41d4-a716-446655440000"
)

for node in dag_status:
    icon = "✅" if node.is_complete else "⚠️"
    print(f"{icon} {node.node_task_id}: {node.completion_percentage:.1f}%")
```

##### `cleanup()`

```python
def cleanup(self)
```

Cleanup resources (thread pool executor).

---

### DAGAssetMapper

Maps DAG nodes to asset specifications via configuration.

```python
from marie.assets import DAGAssetMapper

mapper = DAGAssetMapper(asset_config=config)
```

#### Methods

##### `node_to_asset_specs()`

```python
def node_to_asset_specs(
    self,
    dag_name: str,
    node: Query,
    namespace: str = "marie-ai"
) -> List[AssetSpec]
```

Convert a DAG node to asset specifications.

**Parameters:**
- `dag_name` - Name of the DAG
- `node` - Query node from QueryPlan
- `namespace` - Asset namespace (default: "marie-ai")

**Returns:**
- List of `AssetSpec` for this node

**Example:**

```python
specs = mapper.node_to_asset_specs(
    dag_name="document_processing",
    node=ocr_node
)

for spec in specs:
    print(f"{spec.key}: {spec.kind} (primary={spec.is_primary})")
```

##### `get_upstream_assets_for_node()` (static)

```python
@staticmethod
def get_upstream_assets_for_node(
    dag_id: str,
    node_task_id: str,
    get_connection_fn,
    close_connection_fn
) -> List[Dict[str, str]]
```

Get upstream assets for a DAG node based on actual lineage (what was actually consumed by this node).

**Parameters:**
- `dag_id` - DAG ID
- `node_task_id` - Node task ID
- `get_connection_fn` - Function to get DB connection
- `close_connection_fn` - Function to close DB connection

**Returns:**
- List of dicts with keys: `asset_key`, `latest_version`, `partition_key`

**Example:**

```python
upstream = DAGAssetMapper.get_upstream_assets_for_node(
    dag_id="550e8400-e29b-41d4-a716-446655440000",
    node_task_id="extract",
    get_connection_fn=conn_func,
    close_connection_fn=close_func
)

for asset in upstream:
    print(f"Upstream: {asset['asset_key']} @ {asset['latest_version']}")
```

---

## Data Models

### AssetSpec

Specification for a single asset produced by a node.

```python
@dataclass
class AssetSpec:
    key: str                           # e.g., "ocr/text"
    kind: AssetKind                    # Asset type
    is_primary: bool = True            # Is this the main output?
    is_required: bool = True           # Is this critical?
    description: Optional[str] = None  # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Methods:**
- `to_dict() -> Dict[str, Any]` - Convert to dictionary for JSON serialization

### AssetInfo

Asset registry entry.

```python
@dataclass
class AssetInfo:
    id: int
    asset_key: str
    namespace: str
    kind: str
    description: Optional[str]
    tags: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

### AssetLatestInfo

Latest version information for an asset.

```python
@dataclass
class AssetLatestInfo:
    asset_key: str
    version: str
    latest_at: datetime
    partition_key: Optional[str] = None
```

### MaterializationInfo

Asset materialization event.

```python
@dataclass
class MaterializationInfo:
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
```

### NodeMaterializationStatus

Materialization status for a DAG node.

```python
@dataclass
class NodeMaterializationStatus:
    dag_id: str
    dag_name: str
    node_task_id: str
    expected_assets: int
    materialized_assets: int
    required_assets: int
    materialized_required: int
    all_required_materialized: bool
    missing_required_assets: List[str] = field(default_factory=list)
```

**Properties:**
- `is_complete: bool` - Check if all required assets are materialized
- `completion_percentage: float` - Calculate completion percentage (0-100)

### AssetLineage

Complete lineage information for an asset.

```python
@dataclass
class AssetLineage:
    asset_key: str
    upstream: List[UpstreamAssetInfo] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)
```

### UpstreamAssetInfo

Information about an upstream asset.

```python
@dataclass
class UpstreamAssetInfo:
    asset_key: str
    version: Optional[str]
    partition_key: Optional[str]
```

---

## Enums

### AssetKind

Standard asset types in Marie-AI.

```python
class AssetKind(str, Enum):
    TEXT = "text"                      # Plain text
    JSON = "json"                      # Structured JSON
    BBOX = "bbox"                      # Bounding boxes
    CLASSIFICATION = "classification"  # Classification results
    VECTOR = "vector"                  # Embeddings/vectors
    IMAGE = "image"                    # Images
    TABLE = "table"                    # Tabular data
    METADATA = "metadata"              # Metadata/diagnostics
    BLOB = "blob"                      # Binary blobs
```

---

## Type Aliases

### AssetKey

```python
AssetKey = Tuple[str, str]  # (namespace, path)
```

---

## Predefined Asset Specs

Pre-configured asset specifications for common node types:

### OCR_NODE_ASSETS

```python
OCR_NODE_ASSETS = NodeAssetSpec(
    node_task_id="ocr",
    assets=[
        AssetSpec(
            key="ocr/text",
            kind=AssetKind.TEXT,
            is_primary=True,
            description="Extracted text from OCR"
        ),
        AssetSpec(
            key="ocr/bboxes",
            kind=AssetKind.BBOX,
            is_primary=False,
            description="Bounding boxes for text regions"
        ),
        AssetSpec(
            key="ocr/confidence",
            kind=AssetKind.METADATA,
            is_required=False,
            description="OCR confidence scores"
        ),
    ],
)
```

### EXTRACT_NODE_ASSETS

```python
EXTRACT_NODE_ASSETS = NodeAssetSpec(
    node_task_id="extract",
    assets=[
        AssetSpec(
            key="extract/claims",
            kind=AssetKind.JSON,
            is_primary=True,
            description="Extracted claims data"
        ),
        AssetSpec(
            key="extract/headers",
            kind=AssetKind.JSON,
            is_primary=False,
            description="Extracted header information"
        ),
        AssetSpec(
            key="extract/service_lines",
            kind=AssetKind.TABLE,
            is_primary=False,
            description="Extracted service line items"
        ),
        AssetSpec(
            key="extract/metadata",
            kind=AssetKind.METADATA,
            is_required=False,
            description="Extraction metadata and statistics"
        ),
    ],
)
```

### CLASSIFY_NODE_ASSETS

```python
CLASSIFY_NODE_ASSETS = NodeAssetSpec(
    node_task_id="classify",
    assets=[
        AssetSpec(
            key="classify/document_type",
            kind=AssetKind.CLASSIFICATION,
            is_primary=True,
            description="Document classification result"
        ),
        AssetSpec(
            key="classify/confidence_scores",
            kind=AssetKind.METADATA,
            is_required=False,
            description="Classification confidence scores"
        ),
    ],
)
```

### LOCATE_NODE_ASSETS

```python
LOCATE_NODE_ASSETS = NodeAssetSpec(
    node_task_id="locate",
    assets=[
        AssetSpec(
            key="locate/tables",
            kind=AssetKind.BBOX,
            is_primary=True,
            description="Located table regions"
        ),
        AssetSpec(
            key="locate/forms",
            kind=AssetKind.BBOX,
            is_primary=False,
            description="Located form regions"
        ),
    ],
)
```

### PREDEFINED_ASSET_SPECS

Registry of predefined specs:

```python
PREDEFINED_ASSET_SPECS = {
    "ocr": OCR_NODE_ASSETS,
    "extract": EXTRACT_NODE_ASSETS,
    "classify": CLASSIFY_NODE_ASSETS,
    "locate": LOCATE_NODE_ASSETS,
}
```

**Usage:**

```python
from marie.assets.models import PREDEFINED_ASSET_SPECS

# Get predefined specs
ocr_specs = PREDEFINED_ASSET_SPECS["ocr"]
print(ocr_specs.primary_asset.key)  # "ocr/text"
```

---

## Next Steps

- **[Examples](./examples.md)** - Real-world usage patterns
- **[Usage Guide](./usage.md)** - Integration guide
- **[Configuration Guide](./configuration.md)** - Configuration reference
