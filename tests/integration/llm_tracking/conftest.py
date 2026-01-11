"""
Integration test fixtures for LLM tracking.

Provides fixtures for PostgreSQL, S3 (MinIO), RabbitMQ, and ClickHouse.
Requires Docker Compose services to be running.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest

from marie.llm_tracking.config import configure_from_yaml, reset_settings

# Test configuration values - match docker-compose.yml
TEST_POSTGRES_URL = "postgresql://postgres:test@localhost:5433/llm_tracking_test"
TEST_S3_BUCKET = "llm-tracking-test"
TEST_S3_ENDPOINT = "http://localhost:9002"
TEST_RABBITMQ_URL = "amqp://guest:guest@localhost:5673/"
TEST_CLICKHOUSE_HOST = "localhost"
TEST_CLICKHOUSE_PORT = 8124
TEST_CLICKHOUSE_NATIVE_PORT = 9001


def wait_for_service(check_fn, timeout=60, interval=2, name="service"):
    """Wait for a service to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if check_fn():
                return True
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Timeout waiting for {name} to become available")


@pytest.fixture(scope="session")
def docker_compose():
    """
    Start test infrastructure via Docker Compose.

    This fixture starts all required services and waits for them
    to be healthy before running tests.
    """
    compose_file = Path(__file__).parent / "docker-compose.yml"

    if not compose_file.exists():
        pytest.skip("docker-compose.yml not found")

    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Docker not available")

    # Start services
    subprocess.run(
        ["docker-compose", "-f", str(compose_file), "up", "-d"],
        check=True,
    )

    # Wait for services to be healthy
    try:
        # Wait for PostgreSQL
        def check_postgres():
            import psycopg2

            conn = psycopg2.connect(TEST_POSTGRES_URL)
            conn.close()
            return True

        wait_for_service(check_postgres, name="PostgreSQL")

        # Wait for RabbitMQ
        def check_rabbitmq():
            import pika

            conn = pika.BlockingConnection(
                pika.URLParameters(TEST_RABBITMQ_URL)
            )
            conn.close()
            return True

        wait_for_service(check_rabbitmq, name="RabbitMQ")

        # Wait for MinIO
        def check_minio():
            import urllib.request

            urllib.request.urlopen(f"{TEST_S3_ENDPOINT}/minio/health/live")
            return True

        wait_for_service(check_minio, name="MinIO")

        yield

    finally:
        # Cleanup after all tests (optional - keep services running for faster re-runs)
        # subprocess.run(
        #     ["docker-compose", "-f", str(compose_file), "down", "-v"],
        #     check=False,
        # )
        pass


@pytest.fixture
def llm_tracking_config(docker_compose):
    """
    Configure LLM tracking with test settings.

    Returns the configured settings after setup.
    """
    reset_settings()

    config = {
        "enabled": True,
        "exporter": "rabbitmq",
        "project_id": "test-project",
        "postgres": {
            "url": TEST_POSTGRES_URL,
        },
        "s3": {
            "bucket": TEST_S3_BUCKET,
        },
        "rabbitmq": {
            "url": TEST_RABBITMQ_URL,
            "exchange": "llm-events-test",
            "queue": "llm-ingestion-test",
            "routing_key": "llm.event.test",
        },
        "clickhouse": {
            "host": TEST_CLICKHOUSE_HOST,
            "port": TEST_CLICKHOUSE_PORT,
            "native_port": TEST_CLICKHOUSE_NATIVE_PORT,
            "database": "marie_llm_test",
            "user": "default",
            "password": "",
        },
    }

    # Set environment for S3 (MinIO)
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["AWS_ENDPOINT_URL"] = TEST_S3_ENDPOINT

    settings = configure_from_yaml(config)
    yield settings
    reset_settings()


@pytest.fixture
def postgres_storage(llm_tracking_config) -> Generator:
    """
    PostgresStorage connected to test database.

    Yields a started storage instance, stops it after test.
    """
    from marie.llm_tracking.storage.postgres import PostgresStorage

    storage = PostgresStorage(postgres_url=TEST_POSTGRES_URL)
    storage.start()

    yield storage

    storage.stop()


@pytest.fixture
def s3_storage(llm_tracking_config) -> Generator:
    """
    S3Storage connected to test MinIO.

    Yields a started storage instance, stops it after test.
    """
    from marie.llm_tracking.storage.s3 import S3Storage

    storage = S3Storage(bucket=TEST_S3_BUCKET)
    storage.start()

    yield storage

    storage.stop()


@pytest.fixture
def rabbitmq_exporter(llm_tracking_config) -> Generator:
    """
    RabbitMQExporter connected to test RabbitMQ.

    Yields a started exporter instance, stops it after test.
    """
    from marie.llm_tracking.exporters.rabbitmq import RabbitMQExporter

    exporter = RabbitMQExporter(
        rabbitmq_url=TEST_RABBITMQ_URL,
        exchange="llm-events-test",
        routing_key="llm.event.test",
    )
    exporter.start()

    yield exporter

    exporter.stop()


@pytest.fixture
def clickhouse_writer(llm_tracking_config) -> Generator:
    """
    ClickHouseWriter connected to test ClickHouse.

    Yields a started writer instance, stops it after test.
    """
    from marie.llm_tracking.clickhouse.writer import ClickHouseWriter

    writer = ClickHouseWriter(
        host=TEST_CLICKHOUSE_HOST,
        port=TEST_CLICKHOUSE_NATIVE_PORT,
        database="marie_llm_test",
        user="default",
        password="",
    )
    writer.start()

    yield writer

    writer.shutdown()


@pytest.fixture
def llm_tracker(postgres_storage, s3_storage, rabbitmq_exporter) -> Generator:
    """
    Fully configured LLMTracker for integration tests.

    Yields a started tracker instance, stops it after test.
    """
    from marie.llm_tracking.tracker import LLMTracker, get_tracker

    # Reset singleton
    LLMTracker._instance = None

    tracker = get_tracker()
    tracker.start()

    yield tracker

    tracker.stop()
    LLMTracker._instance = None


@pytest.fixture
def clean_postgres(postgres_storage):
    """
    Clean PostgreSQL tables before and after test.

    Useful for tests that need a clean database state.
    """
    # Clean before test
    conn = postgres_storage._get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {postgres_storage._table_name} CASCADE")
            cur.execute(f"TRUNCATE TABLE {postgres_storage._failed_table_name} CASCADE")
        conn.commit()
    finally:
        postgres_storage._close_connection(conn)

    yield postgres_storage

    # Clean after test
    conn = postgres_storage._get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {postgres_storage._table_name} CASCADE")
            cur.execute(f"TRUNCATE TABLE {postgres_storage._failed_table_name} CASCADE")
        conn.commit()
    finally:
        postgres_storage._close_connection(conn)


# Mark all tests in this directory as integration tests
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
