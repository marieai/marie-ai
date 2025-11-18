"""
Example usage of Marie-AI asset decorators.

This file demonstrates how to use @asset, @multi_asset, and other decorators
to define assets with metadata, dependencies, and configuration.

Note: All asset functions receive an AssetExecutionContext as the first parameter.
This context provides run_id, dag_id, node_task_id, and other runtime information.
"""

from marie.assets import asset, graph_asset, multi_asset, observable_source_asset
from marie.assets.models import AssetKind

# ============================================================================
# Single Asset Examples
# ============================================================================


@asset(
    key="ocr/text",
    kind=AssetKind.TEXT,
    kinds=["craft_ocr", "torch", "python"],
    metadata={"model": "craft-v2", "language": "en"},
    description="Extracted text from documents using CRAFT OCR",
    owners=["ocr-team@company.com"],
    tags={"domain": "ocr", "priority": "high"},
    code_version="1.0.0",
)
def extract_text(context, docs):
    """Extract text from document images."""
    # Implementation here
    extracted_text = "Sample extracted text"
    return extracted_text


@asset(
    key_prefix="classify",
    kind=AssetKind.CLASSIFICATION,
    kinds=["llm", "huggingface", "python"],
    metadata={"model": "document-classifier-v2"},
    description="Document classification results",
    deps=["ocr/text"],  # Depends on text extraction
    group_name="document_processing",
)
def classify_document(context, text):
    """Classify documents based on extracted text."""
    # Implementation here
    classification = {"type": "invoice", "confidence": 0.95}
    return classification


@asset(
    key="extract/claims",
    kind=AssetKind.JSON,
    kinds=["python"],
    deps=["ocr/text", "classify/classify_document"],  # Multiple dependencies
    metadata={"extractor": "claims-v1"},
    is_required=True,
)
def extract_claims(context, text, classification):
    """Extract claims data from classified documents."""
    # Implementation here
    claims = {"claim_id": "12345", "amount": 1000.0}
    return claims


# ============================================================================
# Multi-Asset Examples
# ============================================================================


@multi_asset(
    specs=[
        {
            "key": "ocr/text",
            "kind": "text",
            "is_primary": True,
            "description": "Extracted text",
        },
        {
            "key": "ocr/bboxes",
            "kind": "bbox",
            "is_primary": False,
            "description": "Text bounding boxes",
        },
        {
            "key": "ocr/confidence",
            "kind": "metadata",
            "is_required": False,
            "description": "OCR confidence scores",
        },
    ],
    group_name="ocr",
    code_version="2.0.0",
    owners=["ocr-team@company.com"],
    tags={"stage": "extraction"},
)
def ocr_full_extraction(context, docs):
    """
    Perform complete OCR extraction producing multiple outputs.

    Returns dict with keys matching asset keys.
    """
    # Implementation here
    return {
        "ocr/text": "extracted text",
        "ocr/bboxes": [[0, 0, 100, 20], [0, 25, 150, 45]],
        "ocr/confidence": {"overall": 0.95, "per_word": [0.98, 0.92, 0.96]},
    }


@multi_asset(
    specs=[
        {"key": "extract/claims", "kind": "json", "is_primary": True},
        {"key": "extract/headers", "kind": "json", "is_primary": False},
        {"key": "extract/service_lines", "kind": "table", "is_primary": False},
        {"key": "extract/metadata", "kind": "metadata", "is_required": False},
    ],
    key_prefix="extract",
    deps=["ocr/text", "classify/classify_document"],
    group_name="extraction",
    compute_kind="python",
)
def extract_document_data(context, text, classification):
    """
    Extract structured data from documents.

    Produces multiple related outputs from a single extraction operation.
    """
    # Implementation here
    return {
        "extract/claims": {"claim_id": "12345"},
        "extract/headers": {"patient_name": "John Doe"},
        "extract/service_lines": [{"code": "99213", "amount": 150.0}],
        "extract/metadata": {"extraction_time_ms": 234},
    }


# ============================================================================
# Graph Asset Example
# ============================================================================


@graph_asset(
    key="reports/weekly_summary",
    description="Weekly document processing summary report",
    group_name="reporting",
)
def weekly_summary_asset():
    """
    Composite asset built from multiple internal steps.

    The internal steps are not exposed as separate assets.
    """
    # Compose multiple operations
    data = fetch_weekly_data()
    processed = process_statistics(data)
    report = generate_report(processed)
    return report


def fetch_weekly_data():
    """Internal operation - not an asset."""
    return {"documents_processed": 1000}


def process_statistics(data):
    """Internal operation - not an asset."""
    return {"avg_processing_time": 2.5, "success_rate": 0.98}


def generate_report(stats):
    """Internal operation - not an asset."""
    return {"report": stats}


# ============================================================================
# Observable Source Asset Example
# ============================================================================


@observable_source_asset(
    key="s3/raw_documents",
    description="Raw documents uploaded to S3 bucket",
    metadata={"bucket": "marie-documents", "prefix": "raw/"},
)
def observe_s3_documents(context):
    """
    Observe external S3 bucket for new documents.

    Returns metadata about observed state (not the documents themselves).
    """
    # Check S3 for new documents
    observation = {
        "last_modified": "2025-11-11T15:00:00Z",
        "file_count": 42,
        "total_size_bytes": 1024000,
    }
    return observation


@observable_source_asset(
    key="database/customer_records",
    description="Customer records from external database",
    metadata={"database": "postgres", "table": "customers"},
)
def observe_customer_database(context):
    """
    Observe external customer database.

    Tracks changes without materializing the data.
    """
    observation = {
        "row_count": 10000,
        "last_update": "2025-11-11T14:30:00Z",
        "schema_version": "2.0",
    }
    return observation


# ============================================================================
# Advanced Examples with Technology Kinds
# ============================================================================


@asset(
    key="ml/document_embeddings",
    kind=AssetKind.VECTOR,
    kinds=["torch", "huggingface", "python", "ml_model"],
    metadata={
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
    },
    deps=["ocr/text"],
    group_name="ml_features",
    owners=["ml-team@company.com"],
)
def generate_document_embeddings(context, text):
    """Generate document embeddings using transformer model."""
    # Implementation here
    embeddings = [0.1, 0.2, 0.3]  # Simplified
    return embeddings


@asset(
    key="index/semantic_search",
    kind=AssetKind.VECTOR,
    kinds=["postgres", "pgvector", "python"],
    metadata={"index_type": "ivfflat", "lists": 100},
    deps=["ml/document_embeddings"],
    group_name="search",
)
def build_search_index(context, embeddings):
    """Build semantic search index from embeddings."""
    # Implementation here
    index_metadata = {"indexed_count": 1000}
    return index_metadata


# ============================================================================
# Example with Full Metadata
# ============================================================================


@asset(
    name="premium_extract",  # Override function name
    key="extract/premium_claims",
    kind=AssetKind.JSON,
    kinds=["python", "llm", "openai"],
    description="Premium claims extraction using LLM",
    metadata={
        "model": "gpt-4",
        "temperature": 0.0,
        "max_tokens": 2000,
        "cost_per_call": 0.03,
    },
    deps=["ocr/text", "classify/classify_document"],
    group_name="premium_extraction",
    code_version="3.0.0",
    owners=["extraction-team@company.com", "llm-team@company.com"],
    tags={
        "tier": "premium",
        "sla": "1h",
        "cost": "high",
        "priority": "critical",
    },
    is_required=True,
    compute_kind="llm",
)
def extract_premium_claims_with_llm(context, text, classification):
    """
    Premium claims extraction using GPT-4.

    High-accuracy extraction for critical documents with strict SLA.
    """
    # Implementation here
    claims = {
        "claim_id": "PREM-12345",
        "amount": 50000.0,
        "confidence": 0.99,
        "extraction_method": "llm",
    }
    return claims
