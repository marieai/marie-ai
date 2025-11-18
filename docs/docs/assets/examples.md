---
sidebar_position: 6
---

# Examples

Real-world examples and patterns for asset tracking in Marie-AI.

## Example 1: OCR Pipeline with Multi-Asset Tracking

A complete OCR executor that produces text, bounding boxes, and confidence metadata.

### Configuration

```yaml
asset_config:
  ocr_extractor:
    - "ocr/text"
    - "ocr/bboxes"
    - "ocr/confidence"
```

### Executor Implementation

```python
from marie.executor.mixin import StorageMixin
from marie.assets import AssetTracker
from marie import Executor, requests, DocumentArray
import hashlib
import json

class TextExtractionExecutor(StorageMixin, Executor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.code_version = "git:abcd1234"
        self.model_version = "craft-ocr-v2"

        if self.asset_tracking_enabled:
            self.asset_tracker = AssetTracker(
                storage_handler=self.storage_handler,
                storage_conf=self.storage_conf
            )

    @requests(on='/extract')
    async def extract(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            # Run OCR
            text = self._run_craft_ocr(doc.tensor)
            bboxes = self._detect_text_regions(doc.tensor)
            confidence = self._compute_word_confidences(text, bboxes)

            doc.tags['extracted_text'] = text
            doc.tags['bboxes'] = bboxes
            doc.tags['confidence'] = confidence

        # Persist with asset tracking
        if self.asset_tracking_enabled:
            await self.persist(docs=docs, **kwargs)

        return docs

    async def persist(self, docs, job_id, dag_id=None, node_task_id=None, **kwargs):
        """Record OCR assets: text, bboxes, confidence."""

        assets = []

        for doc in docs:
            text = doc.tags.get('extracted_text', '')
            bboxes = doc.tags.get('bboxes', [])
            confidence = doc.tags.get('confidence', {})

            upstream_versions = self._get_upstream_versions(dag_id, node_task_id)

            # Text asset
            text_bytes = text.encode('utf-8')
            text_version = AssetTracker.compute_asset_version(
                payload_bytes=text_bytes,
                code_fingerprint=self.code_version,
                prompt_fingerprint=self.model_version,
                upstream_versions=upstream_versions
            )

            assets.append({
                "asset_key": "ocr/text",
                "version": text_version,
                "kind": "text",
                "size_bytes": len(text_bytes),
                "checksum": hashlib.sha256(text_bytes).hexdigest(),
                "uri": f"s3://docs/{job_id}/text.txt",
                "metadata": {
                    "language": "en",
                    "char_count": len(text),
                    "word_count": len(text.split())
                }
            })

            # Bounding boxes asset
            bbox_bytes = json.dumps(bboxes).encode('utf-8')
            bbox_version = AssetTracker.compute_asset_version(
                payload_bytes=bbox_bytes,
                code_fingerprint=self.code_version,
                upstream_versions=upstream_versions
            )

            assets.append({
                "asset_key": "ocr/bboxes",
                "version": bbox_version,
                "kind": "bbox",
                "size_bytes": len(bbox_bytes),
                "checksum": hashlib.sha256(bbox_bytes).hexdigest(),
                "uri": f"s3://docs/{job_id}/bboxes.json",
                "metadata": {"bbox_count": len(bboxes)}
            })

            # Confidence asset (optional)
            if confidence:
                conf_bytes = json.dumps(confidence).encode('utf-8')
                conf_version = AssetTracker.compute_asset_version(
                    payload_bytes=conf_bytes,
                    code_fingerprint=self.code_version,
                    upstream_versions=upstream_versions
                )

                assets.append({
                    "asset_key": "ocr/confidence",
                    "version": conf_version,
                    "kind": "metadata",
                    "size_bytes": len(conf_bytes),
                    "checksum": hashlib.sha256(conf_bytes).hexdigest(),
                    "uri": f"s3://docs/{job_id}/confidence.json",
                    "metadata": {
                        "avg_confidence": sum(confidence.values()) / len(confidence)
                    }
                })

        # Record materializations
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

---

## Example 2: Classification with Conditional Assets

A document classifier that only records confidence assets when confidence is high enough.

### Configuration

```yaml
asset_config:
  doc_classifier:
    assets:
      - key: "classify/document_type"
        kind: "classification"
        is_primary: true
        is_required: true
      - key: "classify/confidence_scores"
        kind: "metadata"
        is_required: false
      - key: "classify/alternatives"
        kind: "json"
        is_required: false
```

### Executor Implementation

```python
class DocumentClassificationExecutor(StorageMixin, Executor):

    CONFIDENCE_THRESHOLD = 0.80

    @requests(on='/classify')
    async def classify(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc_type, confidence, alternatives = self._classify(doc)

            doc.tags['document_type'] = doc_type
            doc.tags['confidence'] = confidence
            doc.tags['alternatives'] = alternatives

        if self.asset_tracking_enabled:
            await self.persist(docs=docs, **kwargs)

        return docs

    async def persist(self, docs, job_id, dag_id=None, node_task_id=None, **kwargs):
        """Record classification assets with conditional metadata."""

        assets = []

        for doc in docs:
            doc_type = doc.tags['document_type']
            confidence = doc.tags['confidence']
            alternatives = doc.tags.get('alternatives', [])

            upstream_versions = self._get_upstream_versions(dag_id, node_task_id)

            # Always record primary classification
            result = {"document_type": doc_type, "confidence": confidence}
            result_bytes = json.dumps(result).encode('utf-8')
            version = AssetTracker.compute_asset_version(
                payload_bytes=result_bytes,
                code_fingerprint=self.code_version,
                prompt_fingerprint=self.model_version,
                upstream_versions=upstream_versions
            )

            assets.append({
                "asset_key": "classify/document_type",
                "version": version,
                "kind": "classification",
                "size_bytes": len(result_bytes),
                "checksum": hashlib.sha256(result_bytes).hexdigest(),
                "metadata": {
                    "document_type": doc_type,
                    "confidence": confidence
                }
            })

            # Record confidence scores only if high confidence
            if confidence >= self.CONFIDENCE_THRESHOLD:
                conf_bytes = json.dumps({"confidence": confidence}).encode('utf-8')
                conf_version = AssetTracker.compute_asset_version(
                    payload_bytes=conf_bytes,
                    code_fingerprint=self.code_version,
                    upstream_versions=upstream_versions
                )

                assets.append({
                    "asset_key": "classify/confidence_scores",
                    "version": conf_version,
                    "kind": "metadata",
                    "size_bytes": len(conf_bytes),
                    "checksum": hashlib.sha256(conf_bytes).hexdigest()
                })

            # Record alternatives if multiple candidates
            if len(alternatives) > 1:
                alt_bytes = json.dumps(alternatives).encode('utf-8')
                alt_version = AssetTracker.compute_asset_version(
                    payload_bytes=alt_bytes,
                    code_fingerprint=self.code_version,
                    upstream_versions=upstream_versions
                )

                assets.append({
                    "asset_key": "classify/alternatives",
                    "version": alt_version,
                    "kind": "json",
                    "size_bytes": len(alt_bytes),
                    "checksum": hashlib.sha256(alt_bytes).hexdigest(),
                    "metadata": {"alternative_count": len(alternatives)}
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

---

## Example 3: Claims Extraction with Complex Lineage

An extractor that produces multiple structured outputs with clear upstream dependencies.

### Configuration

```yaml
asset_config:
  claims_extractor:
    assets:
      - key: "extract/claims"
        kind: "json"
        is_primary: true
      - key: "extract/headers"
        kind: "json"
      - key: "extract/service_lines"
        kind: "table"
      - key: "extract/validation_errors"
        kind: "metadata"
        is_required: false
```

### Executor Implementation

```python
class ClaimExtractionExecutor(StorageMixin, Executor):

    @requests(on='/extract')
    async def extract(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            # Extract structured data
            claims = self._extract_claims(doc)
            headers = self._extract_headers(doc)
            service_lines = self._extract_service_lines(doc)
            errors = self._validate_extraction(claims, headers, service_lines)

            doc.tags['claims'] = claims
            doc.tags['headers'] = headers
            doc.tags['service_lines'] = service_lines
            doc.tags['validation_errors'] = errors

        if self.asset_tracking_enabled:
            await self.persist(docs=docs, **kwargs)

        return docs

    async def persist(self, docs, job_id, dag_id=None, node_task_id=None, **kwargs):
        """Record extraction assets with validation metadata."""

        assets = []

        for doc in docs:
            claims = doc.tags.get('claims', [])
            headers = doc.tags.get('headers', {})
            service_lines = doc.tags.get('service_lines', [])
            errors = doc.tags.get('validation_errors', [])

            upstream_versions = self._get_upstream_versions(dag_id, node_task_id)

            # Claims asset (primary)
            claims_bytes = json.dumps(claims).encode('utf-8')
            claims_version = AssetTracker.compute_asset_version(
                payload_bytes=claims_bytes,
                code_fingerprint=self.code_version,
                prompt_fingerprint=self.model_version,
                upstream_versions=upstream_versions
            )

            assets.append({
                "asset_key": "extract/claims",
                "version": claims_version,
                "kind": "json",
                "size_bytes": len(claims_bytes),
                "checksum": hashlib.sha256(claims_bytes).hexdigest(),
                "uri": f"s3://claims/{job_id}/claims.json",
                "metadata": {
                    "claim_count": len(claims),
                    "total_amount": sum(c.get('amount', 0) for c in claims)
                }
            })

            # Headers asset
            headers_bytes = json.dumps(headers).encode('utf-8')
            headers_version = AssetTracker.compute_asset_version(
                payload_bytes=headers_bytes,
                code_fingerprint=self.code_version,
                upstream_versions=upstream_versions
            )

            assets.append({
                "asset_key": "extract/headers",
                "version": headers_version,
                "kind": "json",
                "size_bytes": len(headers_bytes),
                "checksum": hashlib.sha256(headers_bytes).hexdigest(),
                "uri": f"s3://claims/{job_id}/headers.json"
            })

            # Service lines asset
            lines_bytes = json.dumps(service_lines).encode('utf-8')
            lines_version = AssetTracker.compute_asset_version(
                payload_bytes=lines_bytes,
                code_fingerprint=self.code_version,
                upstream_versions=upstream_versions
            )

            assets.append({
                "asset_key": "extract/service_lines",
                "version": lines_version,
                "kind": "table",
                "size_bytes": len(lines_bytes),
                "checksum": hashlib.sha256(lines_bytes).hexdigest(),
                "uri": f"s3://claims/{job_id}/service_lines.json",
                "metadata": {"line_count": len(service_lines)}
            })

            # Validation errors (optional)
            if errors:
                errors_bytes = json.dumps(errors).encode('utf-8')
                errors_version = AssetTracker.compute_asset_version(
                    payload_bytes=errors_bytes,
                    code_fingerprint=self.code_version,
                    upstream_versions=upstream_versions
                )

                assets.append({
                    "asset_key": "extract/validation_errors",
                    "version": errors_version,
                    "kind": "metadata",
                    "size_bytes": len(errors_bytes),
                    "checksum": hashlib.sha256(errors_bytes).hexdigest(),
                    "metadata": {
                        "error_count": len(errors),
                        "severity": "warning" if len(errors) < 5 else "error"
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

---

## Example 4: Monitoring Asset Completeness

Query asset status and send alerts for incomplete nodes.

```python
from marie.assets import AssetRepository

class PipelineMonitor:

    def __init__(self, db_config):
        self.repo = AssetRepository(config=db_config)

    async def check_dag_health(self, dag_id: str):
        """Check asset completeness for a DAG."""

        dag_status = await self.repo.get_dag_status(dag_id)

        incomplete_nodes = []
        for node in dag_status:
            if not node.is_complete:
                incomplete_nodes.append({
                    "node": node.node_task_id,
                    "completion": node.completion_percentage,
                    "missing": node.missing_required_assets
                })

        if incomplete_nodes:
            await self._send_alert({
                "dag_id": dag_id,
                "incomplete_count": len(incomplete_nodes),
                "nodes": incomplete_nodes
            })

        return {
            "complete": len(incomplete_nodes) == 0,
            "incomplete_nodes": incomplete_nodes
        }

    async def get_asset_health_summary(self, hours: int = 24):
        """Get asset materialization success rates."""

        # Query via SQL (AssetRepository can be extended)
        conn = self.repo._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                am.asset_key,
                ar.kind,
                COUNT(DISTINCT am.dag_id) as dag_count,
                COUNT(DISTINCT am.id) as materialization_count,
                MAX(am.created_at) as last_materialized
            FROM marie_scheduler.asset_materialization am
            LEFT JOIN marie_scheduler.asset_registry ar ON ar.asset_key = am.asset_key
            WHERE am.created_at > NOW() - INTERVAL '%s hours'
            GROUP BY am.asset_key, ar.kind
            ORDER BY materialization_count DESC
        """, (hours,))

        results = []
        for row in cursor.fetchall():
            results.append({
                "asset_key": row[0],
                "kind": row[1],
                "dag_count": row[2],
                "materialization_count": row[3],
                "last_materialized": row[4]
            })

        cursor.close()
        conn.close()

        return results
```

---

## Example 5: Impact Analysis Before Code Changes

Check what would be affected by changing an asset definition.

```python
async def analyze_impact(asset_key: str, repo: AssetRepository):
    """Analyze impact of changing an asset."""

    # Get lineage
    lineage = await repo.get_lineage(asset_key)

    print(f"Asset: {asset_key}")
    print(f"Upstream dependencies: {len(lineage.upstream)}")
    for up in lineage.upstream:
        print(f"  ← {up.asset_key}")

    print(f"Downstream consumers: {len(lineage.downstream)}")
    for down in lineage.downstream:
        print(f"  → {down}")

    # Get recent materializations
    history = await repo.get_materialization_history(asset_key, limit=100)

    jobs_affected = len(set(m.job_id for m in history))
    dags_affected = len(set(m.dag_id for m in history if m.dag_id))

    print(f"\nRecent activity (last 100 materializations):")
    print(f"  Jobs affected: {jobs_affected}")
    print(f"  DAGs affected: {dags_affected}")

    # Recommendation
    if len(lineage.downstream) > 10:
        print("\n⚠️  WARNING: This asset has many downstream consumers.")
        print("   Consider creating a new asset version instead of modifying.")
    else:
        print("\n✅ Safe to modify (few downstream dependencies)")

# Usage
await analyze_impact("ocr/text", repo)
```

---

## Example 6: Debugging with Lineage

Trace back through lineage to find the root cause of bad data.

```python
async def debug_bad_asset(
    asset_key: str,
    asset_version: str,
    repo: AssetRepository
):
    """Trace lineage to debug bad data."""

    print(f"Debugging: {asset_key} @ {asset_version}")

    # Get materialization info
    history = await repo.get_materialization_history(asset_key, limit=100)

    mat = None
    for m in history:
        if m.asset_version == asset_version:
            mat = m
            break

    if not mat:
        print("Materialization not found!")
        return

    print(f"Job ID: {mat.job_id}")
    print(f"DAG ID: {mat.dag_id}")
    print(f"Node: {mat.node_task_id}")
    print(f"Created: {mat.created_at}")
    print(f"Size: {mat.size_bytes} bytes")
    print(f"Checksum: {mat.checksum}")

    # Get upstream lineage
    lineage = await repo.get_lineage(asset_key)

    print(f"\nUpstream dependencies:")
    for up in lineage.upstream:
        print(f"  {up.asset_key} @ {up.version}")

        # Recursively check upstream
        up_info = await repo.get_asset_info(up.asset_key)
        if up_info:
            print(f"    Kind: {up_info.kind}")
            print(f"    Namespace: {up_info.namespace}")

    # Check if upstream versions have changed
    print(f"\nChecking if upstream assets are stale...")
    for up in lineage.upstream:
        latest = await repo.get_latest_version(up.asset_key)
        if latest and latest.version != up.version:
            print(f"  ⚠️  {up.asset_key}: using {up.version}, latest is {latest.version}")
        else:
            print(f"  ✅ {up.asset_key}: up to date")

# Usage
await debug_bad_asset(
    asset_key="extract/claims",
    asset_version="v:sha256:abc123...",
    repo=repo
)
```

---

## Example 7: Asset Versioning and Caching

Use content-addressed versions to cache expensive operations.

```python
class CachedExtractor(StorageMixin, Executor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo = AssetRepository(config=self.storage_conf)

    async def extract_with_cache(self, doc, dag_id, node_task_id):
        """Check cache before running expensive extraction."""

        # Compute expected version based on inputs
        upstream_versions = self._get_upstream_versions(dag_id, node_task_id)
        input_bytes = doc.tensor.tobytes()

        expected_version = AssetTracker.compute_asset_version(
            payload_bytes=input_bytes,
            code_fingerprint=self.code_version,
            prompt_fingerprint=self.model_version,
            upstream_versions=upstream_versions
        )

        # Check if this version exists
        history = await self.repo.get_materialization_history(
            "extract/claims",
            limit=1000
        )

        for mat in history:
            if mat.asset_version == expected_version:
                # Cache hit!
                self.logger.info(f"Cache HIT for version {expected_version}")
                return self._load_from_uri(mat.uri)

        # Cache miss - run extraction
        self.logger.info(f"Cache MISS for version {expected_version}")
        result = self._run_extraction(doc)

        # Record new materialization
        await self._persist_result(result, expected_version)

        return result
```

---

## Best Practices Demonstrated

### 1. Always Compute Versions with Upstream Dependencies

```python
# ✅ Good
version = AssetTracker.compute_asset_version(
    payload_bytes=data,
    code_fingerprint=self.code_version,
    upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
)

# ❌ Bad
version = hashlib.sha256(data).hexdigest()  # Ignores code and upstream changes
```

### 2. Use Meaningful Metadata

```python
# ✅ Good
"metadata": {
    "claim_count": len(claims),
    "extraction_time_ms": elapsed,
    "model_version": "llm-extract-v2",
    "confidence": 0.95
}

# ❌ Bad
"metadata": {}
```

### 3. Handle Optional Assets Gracefully

```python
# ✅ Good
if confidence > threshold:
    assets.append({...})

# ❌ Bad
assets.append({...})  # Always include, even if empty/meaningless
```

### 4. Non-Blocking Error Handling

```python
# ✅ Good
try:
    await self.asset_tracker.record_materializations(...)
except Exception as e:
    self.logger.error(f"Asset tracking failed: {e}")
    # Job continues

# ❌ Bad
await self.asset_tracker.record_materializations(...)  # Uncaught, blocks job
```

---

## Next Steps

- **[API Reference](./api-reference.md)** - Complete API documentation
- **[Usage Guide](./usage.md)** - Integration patterns
- **[Configuration Guide](./configuration.md)** - Configuration options
