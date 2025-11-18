---
sidebar_position: 3
---

# Usage Guide

This guide shows how to integrate asset tracking into your Marie-AI workflows, from DAG submission to executor implementation.

## Asset Lifecycle

```
1. DAG Submission
   ↓
   [DAG is submitted to scheduler]

2. Job Execution
   ↓
   [Node produces outputs]
   ↓
   [Executor records materialization via AssetTracker]

3. Query & Monitor
   ↓
   [Query asset status via AssetRepository]
```

## Step 1: Record Materializations in Executors

Executors record asset materializations when they persist outputs.

### Using StorageMixin

The `StorageMixin` provides `persist()` method with built-in asset tracking:

```python
from marie.executor.mixin import StorageMixin
from marie.assets import AssetTracker

class TextExtractionExecutor(StorageMixin, Executor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize asset tracker
        if self.asset_tracking_enabled:
            self.asset_tracker = AssetTracker(
                storage_handler=self.storage_handler,
                storage_conf=self.storage_conf
            )

    @requests(on='/extract')
    async def extract(self, docs: DocumentArray, **kwargs):
        # Perform OCR extraction
        for doc in docs:
            text = self._run_ocr(doc)
            bboxes = self._detect_layout(doc)
            confidence = self._compute_confidence(doc)

            # Store outputs
            doc.tags['extracted_text'] = text
            doc.tags['bboxes'] = bboxes
            doc.tags['confidence'] = confidence

        # Persist with asset tracking
        if self.asset_tracking_enabled:
            await self.persist(
                docs=docs,
                job_id=kwargs.get('job_id'),
                dag_id=kwargs.get('dag_id'),
                node_task_id=kwargs.get('node_task_id'),
                partition_key=kwargs.get('partition_key'),
            )

        return docs

    async def persist(self, docs, job_id, dag_id=None, node_task_id=None, partition_key=None):
        """Persist outputs and record asset materializations."""

        # 1) Save to storage (S3, filesystem, etc.)
        storage_event_id = await self._save_to_storage(docs)

        # 2) Prepare asset list
        assets = []

        for doc in docs:
            text = doc.tags.get('extracted_text', '')
            bboxes = doc.tags.get('bboxes', [])
            confidence = doc.tags.get('confidence', {})

            # Compute version (content-addressed)
            text_version = AssetTracker.compute_asset_version(
                payload_bytes=text.encode('utf-8'),
                code_fingerprint=self.code_version,  # e.g., git commit
                prompt_fingerprint=self.model_version,  # e.g., "craft-ocr-v2"
                upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
            )

            # Add text asset
            assets.append({
                "asset_key": "ocr/text",
                "version": text_version,
                "kind": "text",
                "size_bytes": len(text.encode('utf-8')),
                "checksum": hashlib.sha256(text.encode()).hexdigest(),
                "uri": f"s3://bucket/ocr/{job_id}/text",
                "metadata": {
                    "language": doc.tags.get('language', 'en'),
                    "page_count": doc.tags.get('page_count', 1)
                }
            })

            # Add bbox asset
            bbox_bytes = json.dumps(bboxes).encode('utf-8')
            bbox_version = AssetTracker.compute_asset_version(
                payload_bytes=bbox_bytes,
                code_fingerprint=self.code_version,
                upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
            )

            assets.append({
                "asset_key": "ocr/bboxes",
                "version": bbox_version,
                "kind": "bbox",
                "size_bytes": len(bbox_bytes),
                "checksum": hashlib.sha256(bbox_bytes).hexdigest(),
                "uri": f"s3://bucket/ocr/{job_id}/bboxes",
                "metadata": {"bbox_count": len(bboxes)}
            })

            # Add confidence asset (optional)
            if confidence:
                conf_bytes = json.dumps(confidence).encode('utf-8')
                conf_version = AssetTracker.compute_asset_version(
                    payload_bytes=conf_bytes,
                    code_fingerprint=self.code_version,
                    upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
                )

                assets.append({
                    "asset_key": "ocr/confidence",
                    "version": conf_version,
                    "kind": "metadata",
                    "size_bytes": len(conf_bytes),
                    "checksum": hashlib.sha256(conf_bytes).hexdigest(),
                    "uri": f"s3://bucket/ocr/{job_id}/confidence"
                })

        # 3) Get upstream assets for lineage
        upstream_assets = self._get_upstream_asset_tuples(dag_id, node_task_id)

        # 4) Record materializations
        await self.asset_tracker.record_materializations(
            storage_event_id=storage_event_id,
            assets=assets,
            job_id=job_id,
            dag_id=dag_id,
            node_task_id=node_task_id,
            partition_key=partition_key,
            upstream_assets=upstream_assets,
        )

    def _get_upstream_versions(self, dag_id, node_task_id):
        """Get versions of upstream assets for version computation."""
        if not dag_id or not node_task_id:
            return []

        upstream = DAGAssetMapper.get_upstream_assets_for_node(
            dag_id=dag_id,
            node_task_id=node_task_id,
            get_connection_fn=self.storage_handler._get_connection,
            close_connection_fn=self.storage_handler._close_connection
        )

        return [u['latest_version'] for u in upstream if u['latest_version']]

    def _get_upstream_asset_tuples(self, dag_id, node_task_id):
        """Get upstream asset tuples for lineage recording."""
        if not dag_id or not node_task_id:
            return []

        upstream = DAGAssetMapper.get_upstream_assets_for_node(
            dag_id=dag_id,
            node_task_id=node_task_id,
            get_connection_fn=self.storage_handler._get_connection,
            close_connection_fn=self.storage_handler._close_connection
        )

        return [
            (u['asset_key'], u['latest_version'], u['partition_key'])
            for u in upstream
        ]
```

### Simpler Example (Classification)

For simpler executors with fewer outputs:

```python
from marie.executor.classifier import DocumentClassificationExecutor

class DocumentClassificationExecutor(StorageMixin, Executor):

    @requests(on='/classify')
    async def classify(self, docs: DocumentArray, **kwargs):
        # Perform classification
        for doc in docs:
            doc_type, confidence = self._classify_document(doc)
            doc.tags['document_type'] = doc_type
            doc.tags['confidence'] = confidence

        # Persist with asset tracking
        if self.asset_tracking_enabled:
            await self.persist(docs, **kwargs)

        return docs

    async def persist(self, docs, job_id, dag_id=None, node_task_id=None, **kwargs):
        """Record classification assets."""

        assets = []

        for doc in docs:
            result = {
                "document_type": doc.tags['document_type'],
                "confidence": doc.tags['confidence']
            }
            result_bytes = json.dumps(result).encode('utf-8')

            version = AssetTracker.compute_asset_version(
                payload_bytes=result_bytes,
                code_fingerprint=self.code_version,
                prompt_fingerprint=self.model_version,
                upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
            )

            assets.append({
                "asset_key": "classify/document_type",
                "version": version,
                "kind": "classification",
                "size_bytes": len(result_bytes),
                "checksum": hashlib.sha256(result_bytes).hexdigest(),
                "metadata": {
                    "document_type": result['document_type'],
                    "confidence": result['confidence']
                }
            })

        upstream = self._get_upstream_asset_tuples(dag_id, node_task_id)

        await self.asset_tracker.record_materializations(
            storage_event_id=None,
            assets=assets,
            job_id=job_id,
            dag_id=dag_id,
            node_task_id=node_task_id,
            upstream_assets=upstream,
        )
```

## Step 3: Query Asset Status

Use `AssetRepository` to query asset information and materialization status.

### Initialize Repository

```python
from marie.assets import AssetRepository

# Initialize with database config
repo = AssetRepository(
    config=storage_config,
    max_workers=2
)
```

### Get Asset Information

```python
# Get asset registry entry
asset_info = await repo.get_asset_info("ocr/text")

if asset_info:
    print(f"Asset: {asset_info.asset_key}")
    print(f"Kind: {asset_info.kind}")
    print(f"Namespace: {asset_info.namespace}")
    print(f"Created: {asset_info.created_at}")
```

### Get Latest Version

```python
# Get latest version of an asset
latest = await repo.get_latest_version("ocr/text")

if latest:
    print(f"Latest version: {latest.version}")
    print(f"Materialized at: {latest.latest_at}")
    print(f"Partition: {latest.partition_key}")
```

### Get Materialization History

```python
# Get recent materializations
history = await repo.get_materialization_history(
    asset_key="ocr/text",
    limit=10
)

for mat in history:
    print(f"Version: {mat.asset_version}")
    print(f"Job ID: {mat.job_id}")
    print(f"Created: {mat.created_at}")
    print(f"Size: {mat.size_bytes} bytes")
```

### Get Lineage

```python
# Get upstream dependencies
lineage = await repo.get_lineage("extract/claims")

print(f"Asset: {lineage.asset_key}")
print(f"Upstream dependencies:")
for upstream in lineage.upstream:
    print(f"  - {upstream.asset_key} (v{upstream.version})")
```

### Get Node Materialization Status

```python
# Check if a node has materialized all required assets
status = await repo.get_node_status(
    dag_id="550e8400-e29b-41d4-a716-446655440000",
    node_task_id="ocr_extractor"
)

if status:
    print(f"Node: {status.node_task_id}")
    print(f"Expected assets: {status.expected_assets}")
    print(f"Materialized: {status.materialized_assets}")
    print(f"Completion: {status.completion_percentage:.1f}%")
    print(f"All required complete: {status.is_complete}")

    if not status.is_complete:
        print(f"Missing: {status.missing_required_assets}")
```

### Get DAG-Wide Status

```python
# Get status for all nodes in a DAG
dag_status = await repo.get_dag_status(
    dag_id="550e8400-e29b-41d4-a716-446655440000"
)

for node_status in dag_status:
    completion = node_status.completion_percentage
    icon = "✅" if node_status.is_complete else "⚠️"

    print(f"{icon} {node_status.node_task_id}: {completion:.1f}%")
```

## Step 4: Monitor and Alert

### Integration with Job Supervisor

```python
from marie.job.supervisor import JobSupervisor

class JobSupervisor:

    async def check_asset_completeness(self, dag_id: str):
        """Check if all required assets have been materialized."""

        dag_status = await self.asset_repo.get_dag_status(dag_id)

        incomplete_nodes = [
            s for s in dag_status if not s.is_complete
        ]

        if incomplete_nodes:
            self.logger.warning(
                f"DAG {dag_id} has {len(incomplete_nodes)} nodes with missing required assets"
            )

            for node in incomplete_nodes:
                self.logger.warning(
                    f"  Node {node.node_task_id}: "
                    f"missing {node.missing_required_assets}"
                )

            # Publish alert event
            await self.publish_event({
                "event": "assets.incomplete",
                "dag_id": dag_id,
                "incomplete_nodes": [n.node_task_id for n in incomplete_nodes]
            })
```

### Dashboard Queries

```sql
-- Get assets with highest failure rate
-- Query asset materialization counts by asset key
SELECT
    am.asset_key,
    ar.kind,
    COUNT(DISTINCT am.dag_id) as dag_count,
    COUNT(DISTINCT am.id) as materialization_count,
    MAX(am.created_at) as last_materialized
FROM marie_scheduler.asset_materialization am
LEFT JOIN marie_scheduler.asset_registry ar ON ar.asset_key = am.asset_key
GROUP BY am.asset_key, ar.kind
ORDER BY materialization_count DESC;

-- Get recent asset materializations with lineage
SELECT
    am.asset_key,
    am.asset_version,
    am.job_id,
    am.created_at,
    ARRAY_AGG(al.upstream_asset_key) as upstream_assets
FROM marie_scheduler.asset_materialization am
LEFT JOIN marie_scheduler.asset_lineage al ON al.materialization_id = am.id
WHERE am.created_at > NOW() - INTERVAL '24 hours'
GROUP BY am.id, am.asset_key, am.asset_version, am.job_id, am.created_at
ORDER BY am.created_at DESC
LIMIT 100;
```

## Best Practices

### 1. Enable Asset Tracking Selectively

```python
# Enable for production critical paths
class CriticalExtractor(StorageMixin, Executor):
    asset_tracking_enabled = True

# Disable for development or experimental nodes
class ExperimentalExtractor(Executor):
    asset_tracking_enabled = False
```

### 2. Use Meaningful Asset Keys

```python
# Good ✅
assets = [
    {"asset_key": "ocr/text", ...},
    {"asset_key": "ocr/bboxes", ...}
]

# Bad ❌
assets = [
    {"asset_key": "output1", ...},
    {"asset_key": "result", ...}
]
```

### 3. Include Useful Metadata

```python
assets.append({
    "asset_key": "extract/claims",
    "version": version,
    "metadata": {
        "claim_count": len(claims),
        "total_amount": sum(c.amount for c in claims),
        "extractor_model": "llm-extract-v2",
        "extraction_time_ms": elapsed_ms
    }
})
```

### 4. Handle Errors Gracefully

```python
async def persist(self, docs, **kwargs):
    try:
        # Record assets
        await self.asset_tracker.record_materializations(...)
    except Exception as e:
        # Log but don't fail the job
        self.logger.error(f"Failed to record assets: {e}", exc_info=True)
        # Job continues
```

### 5. Clean Up Resources

```python
def __del__(self):
    if hasattr(self, 'asset_tracker'):
        self.asset_tracker.cleanup()

    if hasattr(self, 'asset_repo'):
        self.asset_repo.cleanup()
```

## Next Steps

- **[CLI Scaffolding](./cli.md)** - Generate executor boilerplate with asset tracking
- **[API Reference](./api-reference.md)** - Detailed API documentation
- **[Examples](./examples.md)** - Real-world patterns and use cases
