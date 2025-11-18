---
sidebar_position: 2
---

# Configuration Guide

Marie-AI's asset tracking uses a **discovery-based approach**—assets are tracked dynamically as they are produced, with **no upfront configuration required**.

## Core Philosophy

Unlike systems that require pre-defining asset schemas (like Dagster), Marie tracks **whatever your executors actually produce**:

```python
# No configuration needed - just emit assets
assets = []
if text_extracted:
    assets.append({"asset_key": "ocr/text", ...})
if tables_detected:
    assets.append({"asset_key": "ocr/tables", ...})

asset_tracker.record_materializations(assets=assets, ...)
```

This is better for document processing because:
- ✅ **Documents are heterogeneous** - PDFs ≠ images ≠ scans
- ✅ **Processing is conditional** - tables only if detected, OCR only if needed
- ✅ **Failures happen** - OCR might fail, classification might be uncertain
- ✅ **Schemas evolve** - new ML models produce different outputs

## Minimal Setup Required

### 1. Enable Asset Tracking (Optional)

Asset tracking is opt-in per executor:

```yaml
# config/service/marie.yml
executors:
  - name: ocr_extractor
    uses: TextExtractionExecutor
    metadata:
      asset_tracking_enabled: true  # Enable for this executor

  - name: fast_classifier
    uses: DocumentClassifierExecutor
    metadata:
      asset_tracking_enabled: false  # Disable for high-throughput nodes
```

**Default:** `true` if PostgreSQL storage is configured.

### 2. Configure PostgreSQL Storage

Assets are stored in PostgreSQL. Configure in your service YAML:

```yaml
shared_config:
  storage:
    psql:
      hostname: ${{ ENV.DB_HOSTNAME }}
      port: 5432
      username: ${{ ENV.DB_USERNAME }}
      password: ${{ ENV.DB_PASSWORD }}
      database: marie
```

Apply the schema:

```bash
psql -U your_user -d your_db -f config/psql/schema/assets.sql
```

### 3. Initialize AssetTracker in Your Executor

```python
from marie.executor.mixin import StorageMixin
from marie.assets import AssetTracker

class MyExecutor(StorageMixin, Executor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # AssetTracker is automatically initialized if asset_tracking_enabled=True
        # Access via self.asset_tracker
```

That's it! No asset schemas to define.

## Asset Naming Conventions

While not required, consistent naming helps with querying and lineage:

### Recommended Pattern: `<domain>/<output_type>`

```python
# Good ✅
"ocr/text"
"ocr/bboxes"
"classify/document_type"
"extract/claims"
"extract/headers"
"index/embeddings"

# Avoid ❌
"output1"
"result"
"data"
"processed"
```

### Common Domains

| Domain | Purpose | Examples |
|--------|---------|----------|
| `ocr/` | Text extraction | `ocr/text`, `ocr/bboxes`, `ocr/confidence` |
| `classify/` | Classification | `classify/document_type`, `classify/language` |
| `extract/` | Structured extraction | `extract/claims`, `extract/tables`, `extract/entities` |
| `locate/` | Layout detection | `locate/regions`, `locate/tables`, `locate/headers` |
| `index/` | Vector embeddings | `index/embeddings`, `index/semantic_chunks` |
| `transform/` | Data transformation | `transform/normalized`, `transform/redacted` |

### Asset Kinds (Auto-Inferred)

The `AssetRegistry` infers kind from asset keys:

| Kind | Key Pattern | Examples |
|------|-------------|----------|
| `text` | `*/text` or ends with `text` | `ocr/text`, `extract_text` |
| `bbox` | Contains `bbox` or `/boxes` | `ocr/bboxes`, `layout/boxes` |
| `classification` | Starts with `classify/` | `classify/document_type` |
| `vector` | Starts with `index/` or contains `embedding` | `index/embeddings` |
| `table` | Contains `table` | `extract/tables` |
| `json` | **Default** | Everything else |

## Executor Implementation

### Basic Pattern

```python
@requests(on='/extract')
async def extract(self, docs: DocumentArray, **kwargs):
    # Extract parameters passed from scheduler
    job_id = kwargs.get('job_id')
    dag_id = kwargs.get('dag_id')
    node_task_id = kwargs.get('node_task_id')

    # Process documents
    results = []
    for doc in docs:
        result = self._process_document(doc)
        results.append(result)

    # Persist with asset tracking (if enabled)
    if self.asset_tracking_enabled:
        self.persist(
            ref_id=doc.id,
            ref_type='document',
            results=results,
            job_id=job_id,
            dag_id=dag_id,
            node_task_id=node_task_id,
        )

    return results
```

### Dynamic Asset Emission

```python
def persist(self, results, job_id, dag_id=None, node_task_id=None):
    """Persist results and dynamically emit assets based on what was produced."""

    assets = []

    for result in results:
        # Always produce text if extraction succeeded
        if result.get('text'):
            assets.append({
                "asset_key": "ocr/text",
                "version": self._compute_version(result['text']),
                "kind": "text",
                "size_bytes": len(result['text'].encode()),
                "metadata": {"char_count": len(result['text'])}
            })

        # Only produce bboxes if layout detection ran
        if result.get('bboxes'):
            assets.append({
                "asset_key": "ocr/bboxes",
                "version": self._compute_version(result['bboxes']),
                "kind": "bbox",
                "metadata": {"bbox_count": len(result['bboxes'])}
            })

        # Only produce tables if any were detected
        if result.get('tables'):
            assets.append({
                "asset_key": "ocr/tables",
                "version": self._compute_version(result['tables']),
                "kind": "table",
                "metadata": {"table_count": len(result['tables'])}
            })

    # Record whatever was actually produced
    if assets:
        self.asset_tracker.record_materializations(
            storage_event_id=None,
            assets=assets,
            job_id=job_id,
            dag_id=dag_id,
            node_task_id=node_task_id,
            upstream_assets=self._get_upstream_asset_tuples(dag_id, node_task_id),
        )
```

## Content-Addressed Versioning

Compute deterministic versions from content + code + upstream:

```python
from marie.assets import AssetTracker

version = AssetTracker.compute_asset_version(
    payload_bytes=result_bytes,
    code_fingerprint=self.code_version,  # e.g., git commit
    prompt_fingerprint=self.model_version,  # e.g., "gpt-4-turbo-2024"
    upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
)
```

This ensures:
- Identical inputs + code → identical version
- Automatic cache invalidation when dependencies change
- Reproducibility across environments

## Querying Assets

### View: Node Materialization Status

```sql
SELECT
  node_task_id,
  materialized_assets,
  asset_keys,
  last_materialized_at
FROM marie_scheduler.node_materialization_status
WHERE dag_id = 'your-dag-id';
```

### Function: Get Upstream Assets

```sql
SELECT * FROM marie_scheduler.get_upstream_assets_for_node(
  'dag-uuid',
  'node-task-id'
);
```

### Direct Queries

```sql
-- All assets produced by a specific job
SELECT asset_key, asset_version, created_at
FROM marie_scheduler.asset_materialization
WHERE job_id = 'job-uuid';

-- All assets of a specific type
SELECT asset_key, COUNT(*) as count
FROM marie_scheduler.asset_materialization am
JOIN marie_scheduler.asset_registry ar ON ar.asset_key = am.asset_key
WHERE ar.kind = 'text'
GROUP BY asset_key
ORDER BY count DESC;
```

## Environment-Specific Configuration

Use environment variables for flexibility:

```yaml
shared_config:
  storage:
    psql:
      hostname: ${{ ENV.DB_HOSTNAME }}
      # Dev: localhost, Prod: db.prod.example.com

executors:
  - name: ocr
    metadata:
      asset_tracking_enabled: ${{ ENV.ENABLE_ASSET_TRACKING }}
      # Dev: false (for speed), Prod: true
```

## Disabling Asset Tracking

### Globally

```yaml
# Disable for all executors
asset_tracking_enabled: false
```

### Per Executor

```yaml
executors:
  - name: fast_ocr
    uses: TextExtractionExecutor
    metadata:
      asset_tracking_enabled: false  # High-throughput, skip tracking
```

### Conditional (in code)

```python
if self.asset_tracking_enabled and not is_test_environment():
    self.persist(...)
```

## Best Practices

### 1. Use Consistent Naming

```python
# Good ✅
assets = [
    {"asset_key": "ocr/text", ...},
    {"asset_key": "ocr/bboxes", ...}
]

# Bad ❌
assets = [
    {"asset_key": f"output_{doc.id}", ...},  # Unique keys break queries
    {"asset_key": "result", ...}  # Too generic
]
```

### 2. Include Useful Metadata

```python
assets.append({
    "asset_key": "extract/claims",
    "version": version,
    "metadata": {
        "claim_count": len(claims),
        "total_amount": sum(c.amount for c in claims),
        "extractor_model": "llm-extract-v2",
        "extraction_time_ms": elapsed_ms,
        "confidence_avg": avg_confidence
    }
})
```

### 3. Handle Errors Gracefully

```python
try:
    assets = self._build_assets(results)
    if assets:
        self.asset_tracker.record_materializations(...)
except Exception as e:
    # Asset tracking failures should NOT break processing
    self.logger.warning(f"Failed to record assets: {e}")
```

### 4. Track Failure Cases

```python
# Even if processing fails, track what happened
if extraction_failed:
    assets.append({
        "asset_key": "extract/error",
        "version": "error",
        "kind": "metadata",
        "metadata": {
            "error_type": str(type(error)),
            "error_message": str(error),
            "failed_at": datetime.utcnow().isoformat()
        }
    })
```

## Next Steps

- **[Usage Guide](./usage.md)** - Implement asset tracking in your executors
- **[Examples](./examples.md)** - Real-world implementation patterns
- **[API Reference](./api-reference.md)** - Complete API documentation
