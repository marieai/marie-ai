"""Fixtures for agent tool unit tests."""
import pytest


@pytest.fixture
def sample_query():
    """Sample query string for testing."""
    return "What is authentication?"


@pytest.fixture
def sample_source_ids():
    """Sample source IDs for testing."""
    return ["source_1", "source_2"]
