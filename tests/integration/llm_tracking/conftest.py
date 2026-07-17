"""
Legacy integration scaffold for instrumentation tests.

The current implementation lives under `marie.instrumentation` and emits
OpenTelemetry/OpenInference telemetry.

This directory currently contains no runnable tests; keep the marker wiring in
place and skip the legacy scaffold until the integration suite is rewritten
against the OTLP collector + ClickHouse path.
"""

from typing import Generator

import pytest

from marie.instrumentation.config import configure, reset_settings
from marie.instrumentation.tracker import LLMTracker, get_tracker

pytestmark = pytest.mark.skip(
    reason=(
        "Legacy llm_tracking integration scaffold targets removed modules. "
        "Rewrite against marie.instrumentation and the OTLP collector pipeline."
    )
)


@pytest.fixture
def instrumentation_config() -> Generator:
    """Configure instrumentation for tests that migrate to the new stack."""
    reset_settings()

    settings = configure(
        {
            "enabled": True,
            "exporter": "otel",
            "project_id": "test-project",
        }
    )

    yield settings
    reset_settings()


@pytest.fixture
def llm_tracker(instrumentation_config) -> Generator:
    """Create a tracker instance for future instrumentation integration tests."""
    LLMTracker._instance = None

    tracker = get_tracker()
    tracker.start()

    yield tracker

    tracker.stop()
    LLMTracker._instance = None


def pytest_configure(config):
    """Configure pytest markers for integration tests."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Docker)"
    )


def pytest_collection_modifyitems(config, items):
    """Add integration marker to all tests in this directory."""
    for item in items:
        if "integration/llm_tracking" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
