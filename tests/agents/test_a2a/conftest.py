"""pytest fixtures for A2A testing.

This module provides fixtures for starting and managing A2A test agents
during integration testing.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Generator

import httpx
import pytest
import pytest_asyncio

# Directory containing test agents
AGENTS_DIR = Path(__file__).parent


@pytest.fixture(scope="session")
def echo_agent() -> Generator[str, None, None]:
    """Start echo agent for tests.

    Yields:
        The base URL of the running echo agent.
    """
    proc = subprocess.Popen(
        [sys.executable, "echo_agent.py"],
        cwd=AGENTS_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = "http://localhost:9001"
    _wait_for_agent(url)

    yield url

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="session")
def calculator_agent() -> Generator[str, None, None]:
    """Start calculator agent for tests.

    Yields:
        The base URL of the running calculator agent.
    """
    proc = subprocess.Popen(
        [sys.executable, "calculator_agent.py"],
        cwd=AGENTS_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = "http://localhost:9002"
    _wait_for_agent(url)

    yield url

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="session")
def streaming_agent() -> Generator[str, None, None]:
    """Start streaming agent for tests.

    Yields:
        The base URL of the running streaming agent.
    """
    proc = subprocess.Popen(
        [sys.executable, "streaming_agent.py"],
        cwd=AGENTS_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = "http://localhost:9003"
    _wait_for_agent(url)

    yield url

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="session")
def async_task_agent() -> Generator[str, None, None]:
    """Start async task agent for tests.

    Yields:
        The base URL of the running async task agent.
    """
    proc = subprocess.Popen(
        [sys.executable, "async_task_agent.py"],
        cwd=AGENTS_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = "http://localhost:9004"
    _wait_for_agent(url)

    yield url

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="session")
def all_test_agents(
    echo_agent: str,
    calculator_agent: str,
    streaming_agent: str,
    async_task_agent: str,
) -> dict[str, str]:
    """Start all test agents.

    Yields:
        Dictionary mapping agent names to their base URLs.
    """
    return {
        "echo": echo_agent,
        "calculator": calculator_agent,
        "streaming": streaming_agent,
        "async": async_task_agent,
    }


@pytest.fixture(scope="session")
def docker_compose_agents() -> Generator[dict[str, str], None, None]:
    """Start all test agents via docker-compose.

    Yields:
        Dictionary mapping agent names to their base URLs.
    """
    compose_file = AGENTS_DIR / "docker-compose.yml"
    if not compose_file.exists():
        pytest.skip("docker-compose.yml not found")

    subprocess.run(
        ["docker-compose", "-f", str(compose_file), "up", "-d", "--build"],
        check=True,
        cwd=AGENTS_DIR,
    )

    # Wait for all agents to be ready
    agents = {
        "echo": "http://localhost:9001",
        "calculator": "http://localhost:9002",
        "streaming": "http://localhost:9003",
        "async": "http://localhost:9004",
    }

    for url in agents.values():
        _wait_for_agent(url, timeout=60)

    yield agents

    subprocess.run(
        ["docker-compose", "-f", str(compose_file), "down"],
        check=True,
        cwd=AGENTS_DIR,
    )


@pytest_asyncio.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client for tests."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


def _wait_for_agent(url: str, timeout: int = 30) -> None:
    """Wait for an agent to become available.

    Args:
        url: Base URL of the agent.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If agent doesn't respond within timeout.
    """
    card_url = f"{url}/.well-known/agent.json"
    start = time.time()

    while time.time() - start < timeout:
        try:
            resp = httpx.get(card_url, timeout=2.0)
            if resp.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.5)

    raise TimeoutError(f"Agent at {url} did not become available within {timeout}s")
