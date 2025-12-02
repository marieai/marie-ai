---
sidebar_position: 4
---

# CLI Scaffolding

The Marie CLI provides scaffolding commands to generate asset tracking boilerplate for executors.

## Quick Start

```bash
# Generate asset tracking code for an executor
marie scaffold asset marie/executor/text/text_extraction_executor.py

# Generate with custom template
marie scaffold asset --template multi-asset marie/executor/custom_executor.py

# Preview without writing
marie scaffold asset --dry-run marie/executor/ocr_executor.py
```

## Basic Usage

### Scaffold Single Asset

```bash
marie scaffold asset path/to/executor.py
```

This generates asset tracking boilerplate for your executor:

```python
# Auto-generated asset tracking code for MyExecutor
# Customize the asset keys and metadata as needed

from marie.executor.mixin import StorageMixin
from marie.serve.executors import Executor, requests
from marie.assets import AssetTracker
import hashlib
import json

class MyExecutor(StorageMixin, Executor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize asset tracker
        if self.asset_tracking_enabled:
            from marie.assets import AssetTracker
            self.asset_tracker = AssetTracker(
                storage_handler=self.storage_handler,
                storage_conf=self.storage_conf
            )

    @requests(on='/process')
    async def process(self, docs: DocumentArray, **kwargs):
        # Your processing logic here
        results = self._do_work(docs)

        # Persist with asset tracking
        if self.asset_tracking_enabled:
            await self.persist(
                docs=docs,
                results=results,
                job_id=kwargs.get('job_id'),
                dag_id=kwargs.get('dag_id'),
                node_task_id=kwargs.get('node_task_id'),
            )

        return docs

    async def persist(self, docs, results, job_id, dag_id=None, node_task_id=None):
        """Persist results and record asset materializations."""

        # Prepare asset list
        assets = []

        for doc, result in zip(docs, results):
            result_bytes = json.dumps(result).encode('utf-8')

            # Compute content-addressed version
            version = AssetTracker.compute_asset_version(
                payload_bytes=result_bytes,
                code_fingerprint=self.code_version,
                prompt_fingerprint=getattr(self, 'model_version', None),
                upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
            )

            assets.append({
                "asset_key": "my_executor/output",
                "version": version,
                "kind": "json",
                "size_bytes": len(result_bytes),
                "checksum": hashlib.sha256(result_bytes).hexdigest(),
                "uri": f"s3://bucket/{job_id}/output",
                "metadata": {}  # Add useful metadata here
            })

        # Get upstream assets for lineage
        upstream = self._get_upstream_asset_tuples(dag_id, node_task_id)

        # Record materializations
        await self.asset_tracker.record_materializations(
            storage_event_id=None,
            assets=assets,
            job_id=job_id,
            dag_id=dag_id,
            node_task_id=node_task_id,
            upstream_assets=upstream,
        )

    def _get_upstream_versions(self, dag_id, node_task_id):
        """Get versions of upstream assets for version computation."""
        if not dag_id or not node_task_id:
            return []

        from marie.assets import DAGAssetMapper

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

        from marie.assets import DAGAssetMapper

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

### Scaffold Multi-Asset

```bash
marie scaffold asset --template multi-asset path/to/executor.py
```

This generates boilerplate for tracking multiple assets per execution:

```python
# Multi-asset tracking example
# Generates primary, secondary, and optional metadata assets

class MyExecutor(StorageMixin, Executor):

    async def persist(self, docs, results, job_id, dag_id=None, node_task_id=None):
        """Persist results and record asset materializations."""

        assets = []

        for doc in docs:
            # Primary asset
            primary_data = doc.tags.get('primary_output')
            primary_bytes = json.dumps(primary_data).encode('utf-8')
            primary_version = AssetTracker.compute_asset_version(
                payload_bytes=primary_bytes,
                code_fingerprint=self.code_version,
                upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
            )

            assets.append({
                "asset_key": "my_executor/primary",
                "version": primary_version,
                "kind": "json",
                "size_bytes": len(primary_bytes),
                "checksum": hashlib.sha256(primary_bytes).hexdigest(),
                "uri": f"s3://bucket/{job_id}/primary"
            })

            # Secondary asset
            secondary_data = doc.tags.get('secondary_output')
            secondary_bytes = json.dumps(secondary_data).encode('utf-8')
            secondary_version = AssetTracker.compute_asset_version(
                payload_bytes=secondary_bytes,
                code_fingerprint=self.code_version,
                upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
            )

            assets.append({
                "asset_key": "my_executor/secondary",
                "version": secondary_version,
                "kind": "json",
                "size_bytes": len(secondary_bytes),
                "checksum": hashlib.sha256(secondary_bytes).hexdigest(),
                "uri": f"s3://bucket/{job_id}/secondary"
            })

            # Optional metadata asset
            if doc.tags.get('metadata'):
                metadata = doc.tags['metadata']
                metadata_bytes = json.dumps(metadata).encode('utf-8')
                metadata_version = AssetTracker.compute_asset_version(
                    payload_bytes=metadata_bytes,
                    code_fingerprint=self.code_version,
                    upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
                )

                assets.append({
                    "asset_key": "my_executor/metadata",
                    "version": metadata_version,
                    "kind": "metadata",
                    "size_bytes": len(metadata_bytes),
                    "checksum": hashlib.sha256(metadata_bytes).hexdigest(),
                    "uri": f"s3://bucket/{job_id}/metadata"
                })

        # Record all materializations
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

## Available Templates

### `single` (default)

Generates configuration for a single primary asset.

```bash
marie scaffold asset path/to/executor.py
# or explicitly:
marie scaffold asset --template single path/to/executor.py
```

**Use when:**
- Executor produces one main output
- Simple use case
- Getting started with asset tracking

### `multi-asset`

Generates configuration for multiple assets (primary, secondary, optional).

```bash
marie scaffold asset --template multi-asset path/to/executor.py
```

**Use when:**
- Executor produces multiple outputs
- Need to track primary and secondary results
- Have optional metadata or diagnostics

### `ocr`

Specialized template for OCR executors with text, bboxes, and confidence.

```bash
marie scaffold asset --template ocr path/to/ocr_executor.py
```

**Use when:**
- Building OCR/text extraction executors
- Need standard OCR outputs (text + layout)

### `classification`

Template for classification executors with predictions and confidence scores.

```bash
marie scaffold asset --template classification path/to/classifier.py
```

**Use when:**
- Building document classifiers
- Need to track predictions with confidence

### `extraction`

Template for data extraction executors (claims, headers, tables).

```bash
marie scaffold asset --template extraction path/to/extractor.py
```

**Use when:**
- Building structured data extractors
- Need to track multiple extraction outputs

## Command Options

### `--template, -t`

Specify which template to use:

```bash
marie scaffold asset --template multi-asset path/to/executor.py
```

### `--output, -o`

Write to a specific output file:

```bash
marie scaffold asset -o generated_executor.py path/to/executor.py
```

### `--dry-run`

Preview the generated code without writing:

```bash
marie scaffold asset --dry-run path/to/executor.py
```

### `--force, -f`

Overwrite existing file:

```bash
marie scaffold asset --force path/to/executor.py
```

### `--config, -c`

Generate YAML configuration instead of Python code:

```bash
marie scaffold asset --config path/to/executor.py

# Outputs:
# asset_config:
#   my_executor:
#     - "my_executor/output"
```

### `--verbose, -v`

Show detailed generation information:

```bash
marie scaffold asset --verbose path/to/executor.py
```

## Configuration File Generation

Generate YAML configuration that can be added to your service config:

```bash
marie scaffold asset --config path/to/executor.py > config/assets.yml
```

**Output (`config/assets.yml`):**

```yaml
# Auto-generated asset configuration
asset_config:
  my_executor:
    assets:
      - key: "my_executor/output"
        kind: "json"
        is_primary: true
        is_required: true
        description: "TODO: Describe this asset"
```

## Batch Scaffolding

Scaffold multiple executors at once:

```bash
# Scaffold all executors in a directory
find marie/executor -name "*_executor.py" | xargs -I {} marie scaffold asset {}

# With specific template
find marie/executor/text -name "*.py" | xargs -I {} marie scaffold asset --template ocr {}
```

## Integration with Existing Code

The scaffold command can detect existing executors and add asset tracking:

```bash
# Given an existing executor:
class MyExecutor(Executor):
    @requests(on='/process')
    async def process(self, docs: DocumentArray, **kwargs):
        return self._do_work(docs)

# Running scaffold:
marie scaffold asset path/to/my_executor.py

# Adds asset tracking without breaking existing code:
class MyExecutor(StorageMixin, Executor):  # Added StorageMixin

    def __init__(self, *args, **kwargs):  # Added __init__
        super().__init__(*args, **kwargs)
        if self.asset_tracking_enabled:
            from marie.assets import AssetTracker
            self.asset_tracker = AssetTracker(
                storage_handler=self.storage_handler,
                storage_conf=self.storage_conf
            )

    @requests(on='/process')
    async def process(self, docs: DocumentArray, **kwargs):
        results = self._do_work(docs)

        # Added asset tracking
        if self.asset_tracking_enabled:
            await self.persist(docs=docs, results=results, **kwargs)

        return results

    async def persist(self, docs, results, **kwargs):  # Added persist method
        # ... generated asset tracking code ...
```

## Customizing Generated Code

After scaffolding, customize these sections:

### 1. Asset Keys

```python
# Generated:
"asset_key": "my_executor/output"

# Customize to:
"asset_key": "ocr/text"  # More descriptive
```

### 2. Asset Metadata

```python
# Generated:
"metadata": {}

# Add useful context:
"metadata": {
    "language": doc.tags.get('language'),
    "confidence": doc.tags.get('confidence'),
    "model_version": self.model_version
}
```

### 3. Conditional Assets

```python
# Add logic for optional assets:
if doc.tags.get('has_tables'):
    assets.append({
        "asset_key": "extract/tables",
        # ...
    })
```

### 4. Error Handling

```python
# Add try-except around asset tracking:
try:
    await self.asset_tracker.record_materializations(...)
except Exception as e:
    self.logger.error(f"Asset tracking failed: {e}", exc_info=True)
    # Continue execution
```

## Best Practices

### 1. Start with Simple Template

Begin with the `single` template and evolve to `multi-asset` as needed.

### 2. Use Descriptive Asset Keys

```bash
# Good
marie scaffold asset --template ocr marie/executor/text/craft_ocr.py
# Generates: "ocr/text", "ocr/bboxes"

# Bad
marie scaffold asset marie/executor/processor.py
# Generates: "processor/output"  # Too generic
```

### 3. Review and Customize

Generated code is a starting point. Always review and customize:
- Asset keys (make them semantic)
- Metadata (add business context)
- Conditional logic (for optional assets)
- Error handling (non-blocking failures)

### 4. Version Control

Commit generated files to track changes:

```bash
marie scaffold asset marie/executor/new_executor.py
git add marie/executor/new_executor.py
git commit -m "feat: add asset tracking to new_executor"
```

### 5. Test Generated Code

Write tests for asset tracking:

```python
def test_asset_tracking():
    executor = MyExecutor(asset_tracking_enabled=True)

    docs = DocumentArray([Document()])
    await executor.process(docs, job_id="test-job", dag_id="test-dag")

    # Verify assets were recorded
    repo = AssetRepository(config=db_config)
    history = await repo.get_materialization_history("my_executor/output", limit=1)

    assert len(history) == 1
    assert history[0].job_id == "test-job"
```

## Troubleshooting

### Scaffold command not found

Ensure Marie CLI is properly installed:

```bash
pip install -e .
marie --version
```

### Generated code has syntax errors

Report the issue with:

```bash
marie scaffold asset --verbose path/to/executor.py > output.log 2>&1
```

### Template not working for my use case

Create a custom template or manually adapt generated code. See [Examples](./examples.md) for patterns.

## Next Steps

- **[API Reference](./api-reference.md)** - Detailed API documentation
- **[Examples](./examples.md)** - Real-world patterns and use cases
- **[Usage Guide](./usage.md)** - Manual integration without scaffolding
