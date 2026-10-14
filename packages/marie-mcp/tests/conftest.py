"""Isolate package tests from workstation credentials and dotenv files."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Source-only package tests must not import the heavyweight host package.
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
with (
    patch('dotenv.load_dotenv', return_value=False),
    patch.dict(
        os.environ,
        {
            'MARIE_API_KEY': 'test-api-key',
            'MARIE_RETRY_MIN_WAIT': '0',
            'MARIE_RETRY_MAX_WAIT': '0',
            'AWS_ACCESS_KEY_ID': '',
            'AWS_SECRET_ACCESS_KEY': '',
        },
    ),
):
    from marie.mcp.server.config import Config  # noqa: F401


@pytest.fixture
def mock_marie_client():
    from marie.mcp.server.clients.marie_client import MarieClient

    client = AsyncMock(spec=MarieClient)
    client.api_key = 'test-api-key'
    client.submit_job.return_value = {'job_id': 'test-job'}
    return client
