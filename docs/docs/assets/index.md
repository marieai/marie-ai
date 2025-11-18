---
sidebar_position: 1
---

# Asset Tracking Overview

Marie-AI's asset tracking system transforms data lineage from passive observation into **strategic capability**. It provides DAG-aware visibility into what your document processing workflows produce, where it comes from, and who depends on it.

## Why Asset Tracking Matters

In modern document intelligence workflows, data doesn't just flow—it transforms, enriches, and compounds. A single document might produce:

- Extracted text (OCR output)
- Bounding boxes (layout detection)
- Classification results (document type)
- Named entities (extracted claims, headers)
- Embeddings (semantic vectors)
- Structured JSON (final output)

**Without lineage**, when something breaks, you're left guessing:
- Which upstream change caused this failure?
- Who owns the logic that produced this asset?
- What downstream systems depend on this data?
- Can I safely change this transformation?

**With asset tracking**, you have:
- ✅ **Trust**: Content-addressed versioning ensures reproducibility
- ✅ **Accountability**: Explicit ownership for every asset produced
- ✅ **Impact analysis**: Understand downstream consequences before making changes
- ✅ **Cross-team visibility**: Shared context across engineering, analytics, and ML teams
- ✅ **AI-readiness**: Feature provenance for ML models and agent workflows

## Asset Tracking ≠ Job Execution Control

**Critical distinction**: Assets track *what was produced* for observability and debugging. They do **not** control job execution. The scheduler manages job dependencies and execution order.

This separation ensures:
- Jobs run based on DAG topology, not asset availability
- Asset tracking is opt-in and non-blocking
- System remains resilient even if asset tracking fails

## Discovery-Based, Not Schema-Driven

Unlike systems that require **pre-defining asset schemas** (like Dagster), Marie uses a **discovery-based approach**:

```python
# ❌ Dagster: Must declare upfront
@multi_asset(outs={"text": AssetOut(), "bboxes": AssetOut(), "tables": AssetOut()})
def process_document(doc):
    # What if there are no tables? What if OCR fails?
    ...

# ✅ Marie: Discover as you produce
def process_document(doc):
    assets = []
    if text_extracted:
        assets.append({"asset_key": "ocr/text", ...})
    if has_tables:
        assets.append({"asset_key": "ocr/tables", ...})

    asset_tracker.record_materializations(assets=assets)
```

**Why this matters for document processing:**
- 📄 **Documents are heterogeneous** - PDFs ≠ images ≠ scans
- 🔀 **Processing is conditional** - tables only if detected, OCR only if needed
- ⚠️ **Failures happen** - OCR might fail, extraction might be uncertain
- 🔄 **Schemas evolve** - new ML models produce different outputs

You **can't know** at definition time what assets a document will produce—you discover them at runtime.

## Core Concepts

### Assets

An **asset** is a named, versioned output produced by a DAG node. Examples:
- `ocr/text` - Extracted text from OCR
- `extract/claims` - Structured claim data
- `classify/document_type` - Classification result
- `index/embeddings` - Vector embeddings

Each asset has:
- **Key**: Unique identifier (e.g., `ocr/text`)
- **Kind**: Type classification (TEXT, JSON, BBOX, VECTOR, etc.)
- **Version**: Content-addressed hash (deterministic)
- **Metadata**: Size, checksum, URI, custom properties
- **Lineage**: Upstream dependencies (what it was built from)

### Multi-Asset Nodes

A single DAG node can produce **multiple assets**. For example, an OCR node might output:
- `ocr/text` (primary, required)
- `ocr/bboxes` (secondary, required)
- `ocr/confidence` (optional metadata)

This enables fine-grained tracking without artificially splitting logic across nodes.

### Content-Addressed Versioning

Asset versions are **deterministic hashes** computed from:
- Payload content (the actual data)
- Code fingerprint (git commit, deployment version)
- Prompt fingerprint (model version for LLM-based extraction)
- Upstream asset versions (dependencies)

This ensures:
- Identical inputs + code = identical version
- Automatic cache invalidation when dependencies change
- Reproducibility across environments

### Lineage

**Lineage** tracks the dependency graph between assets. It's automatically inferred from DAG topology:

```
upstream_asset_1 ──┐
                   ├──> node_transform ──> downstream_asset
upstream_asset_2 ──┘
```

Lineage enables:
- Root cause analysis (which upstream change broke this?)
- Impact analysis (what breaks if I change this?)
- Audit trails (how was this asset produced?)
- Feature provenance (for ML models)

## When to Use Asset Tracking

✅ **Use asset tracking when:**
- You need observability into what your DAG produces
- Multiple teams consume outputs and need to understand dependencies
- Compliance requires audit trails of data transformations
- You're building ML models and need feature provenance
- You want to understand impact before making changes

❌ **Don't use asset tracking for:**
- Controlling job execution (use DAG dependencies instead)
- Real-time decision making (assets are recorded after-the-fact)
- Replacing your scheduler's dependency graph

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   DAG Execution                      │
│                                                      │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐ │
│  │   OCR    │─────▶│ Classify │─────▶│ Extract  │ │
│  └──────────┘      └──────────┘      └──────────┘ │
│       │                  │                  │       │
│       │ produce          │ produce          │ produce
│       ▼                  ▼                  ▼       │
│  [ocr/text]     [classify/type]      [extract/claims]
│  [ocr/bboxes]   [classify/confidence] [extract/headers]
└─────────────────────────────────────────────────────┘
                           │
                           │ record
                           ▼
                ┌──────────────────────┐
                │   AssetTracker       │
                │                      │
                │  • Materialization   │
                │  • Lineage           │
                │  • Versioning        │
                └──────────────────────┘
                           │
                           │ persist
                           ▼
                ┌──────────────────────┐
                │  PostgreSQL          │
                │                      │
                │  • asset_registry    │
                │  • asset_materialization
                │  • asset_lineage     │
                │  • asset_latest      │
                └──────────────────────┘
                           │
                           │ query
                           ▼
                ┌──────────────────────┐
                │  AssetRepository     │
                │                      │
                │  • get_asset_info()  │
                │  • get_lineage()     │
                │  • get_node_status() │
                └──────────────────────┘
```

## Next Steps

- **[Configuration Guide](./configuration.md)** - Set up asset tracking with minimal YAML
- **[Usage Guide](./usage.md)** - Integrate asset tracking into your executors
- **[CLI Scaffolding](./cli.md)** - Generate asset definitions from templates
- **[API Reference](./api-reference.md)** - Detailed API documentation
- **[Examples](./examples.md)** - Real-world use cases and patterns
