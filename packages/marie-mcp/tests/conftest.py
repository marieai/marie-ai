"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def mock_marie_client():
    """Mock Marie client for testing."""
    from unittest.mock import AsyncMock

    from marie_mcp.clients.marie_client import MarieClient

    client = AsyncMock(spec=MarieClient)
    client.api_key = "test-api-key"
    return client


@pytest.fixture
def sample_job_response():
    """Sample job submission response."""
    return {"job_id": "test_job_123", "status": "submitted"}


@pytest.fixture
def sample_metadata():
    """Sample job metadata."""
    return {
        "on": "extract_executor://document/extract",
        "project_id": "test-project",
        "ref_id": "test_doc_001",
        "ref_type": "invoice",
        "uri": "s3://marie/invoice/test_doc_001/test.pdf",
        "policy": "allow_all",
        "planner": "extract",
        "type": "pipeline",
        "name": "default",
    }
