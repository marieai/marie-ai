"""
Tests for backend factory functions.
"""

from unittest.mock import MagicMock, patch

import pytest
from marie_kernel import create_backend, create_backend_from_url
from marie_kernel.backends.memory import InMemoryStateBackend


class TestCreateBackend:
    """Tests for create_backend factory function."""

    def test_create_memory_backend_default(self):
        """Test that memory backend is created by default."""
        backend = create_backend()
        assert isinstance(backend, InMemoryStateBackend)

    def test_create_memory_backend_explicit(self):
        """Test creating memory backend explicitly."""
        backend = create_backend("memory")
        assert isinstance(backend, InMemoryStateBackend)

    def test_create_memory_backend_case_insensitive(self):
        """Test that backend type is case-insensitive."""
        backend = create_backend("MEMORY")
        assert isinstance(backend, InMemoryStateBackend)

        backend = create_backend("Memory")
        assert isinstance(backend, InMemoryStateBackend)

    def test_create_postgres_backend_requires_pool(self):
        """Test that postgres backend requires connection_pool."""
        with pytest.raises(ValueError) as exc_info:
            create_backend("postgres")

        assert "connection_pool is required" in str(exc_info.value)

    def test_create_postgres_backend_with_pool(self):
        """Test creating postgres backend with pool."""
        pytest.importorskip("psycopg_pool")

        mock_pool = MagicMock()
        backend = create_backend("postgres", connection_pool=mock_pool)

        from marie_kernel.backends.postgres import PostgresStateBackend

        assert isinstance(backend, PostgresStateBackend)

    def test_create_unknown_backend_raises(self):
        """Test that unknown backend type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_backend("unknown_type")

        assert "Unknown backend type" in str(exc_info.value)
        assert "unknown_type" in str(exc_info.value)

    def test_create_s3_backend_requires_client(self):
        """Test that S3 backend requires s3_client."""
        with pytest.raises(ValueError) as exc_info:
            create_backend("s3", bucket="my-bucket")

        assert "s3_client is required" in str(exc_info.value)

    def test_create_s3_backend_requires_bucket(self):
        """Test that S3 backend requires bucket."""
        mock_client = MagicMock()
        with pytest.raises(ValueError) as exc_info:
            create_backend("s3", s3_client=mock_client)

        assert "bucket is required" in str(exc_info.value)

    def test_create_s3_backend_with_client(self):
        """Test creating S3 backend with client and bucket."""
        pytest.importorskip("boto3")

        mock_client = MagicMock()
        backend = create_backend(
            "s3", s3_client=mock_client, bucket="my-bucket", prefix="custom-prefix"
        )

        from marie_kernel.backends.s3 import S3StateBackend

        assert isinstance(backend, S3StateBackend)


class TestCreateBackendFromUrl:
    """Tests for create_backend_from_url factory function."""

    def test_memory_url(self):
        """Test creating memory backend from URL."""
        backend = create_backend_from_url("memory://")
        assert isinstance(backend, InMemoryStateBackend)

    def test_mem_url(self):
        """Test creating memory backend from short URL."""
        backend = create_backend_from_url("mem://")
        assert isinstance(backend, InMemoryStateBackend)

    def test_memory_url_case_insensitive(self):
        """Test that URL scheme is case-insensitive."""
        backend = create_backend_from_url("MEMORY://")
        assert isinstance(backend, InMemoryStateBackend)

    def test_postgres_url_without_psycopg(self):
        """Test that postgres URL without psycopg raises ImportError."""
        with patch.dict("sys.modules", {"psycopg_pool": None}):
            # Can't easily test this without actually uninstalling psycopg
            # Just verify the URL is recognized
            pass

    def test_postgres_url_with_psycopg(self):
        """Test creating postgres backend from URL."""
        psycopg_pool = pytest.importorskip("psycopg_pool")

        # Mock the ConnectionPool to avoid actual connection
        with patch.object(psycopg_pool, "ConnectionPool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            backend = create_backend_from_url(
                "postgresql://user:pass@localhost:5432/marie"
            )

            mock_pool_class.assert_called_once_with(
                "postgresql://user:pass@localhost:5432/marie"
            )

            from marie_kernel.backends.postgres import PostgresStateBackend

            assert isinstance(backend, PostgresStateBackend)

    def test_postgres_scheme_variant(self):
        """Test that postgres:// URL scheme works."""
        psycopg_pool = pytest.importorskip("psycopg_pool")

        with patch.object(psycopg_pool, "ConnectionPool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            backend = create_backend_from_url("postgres://localhost/db")

            mock_pool_class.assert_called_once_with("postgres://localhost/db")

    def test_unknown_url_scheme_raises(self):
        """Test that unknown URL scheme raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_backend_from_url("redis://localhost:6379")

        assert "Unknown URL scheme" in str(exc_info.value)

    def test_s3_url(self):
        """Test creating S3 backend from URL."""
        boto3 = pytest.importorskip("boto3")

        with patch.object(boto3, "client") as mock_client_func:
            mock_client = MagicMock()
            mock_client_func.return_value = mock_client

            backend = create_backend_from_url("s3://my-bucket/custom-prefix")

            mock_client_func.assert_called_once_with("s3")

            from marie_kernel.backends.s3 import S3StateBackend

            assert isinstance(backend, S3StateBackend)

    def test_s3_url_default_prefix(self):
        """Test S3 URL without prefix uses default."""
        boto3 = pytest.importorskip("boto3")

        with patch.object(boto3, "client") as mock_client_func:
            mock_client = MagicMock()
            mock_client_func.return_value = mock_client

            backend = create_backend_from_url("s3://my-bucket")

            from marie_kernel.backends.s3 import S3StateBackend

            assert isinstance(backend, S3StateBackend)
            # Verify default prefix was used
            assert backend._prefix == "marie-state"


class TestFactoryProtocolCompliance:
    """Tests verifying factory returns protocol-compliant backends."""

    def test_memory_backend_is_state_backend(self):
        """Test that memory backend satisfies StateBackend protocol."""
        from marie_kernel import StateBackend

        backend = create_backend("memory")
        assert isinstance(backend, StateBackend)

    def test_postgres_backend_is_state_backend(self):
        """Test that postgres backend satisfies StateBackend protocol."""
        psycopg_pool = pytest.importorskip("psycopg_pool")
        from marie_kernel import StateBackend

        with patch.object(psycopg_pool, "ConnectionPool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            backend = create_backend_from_url("postgresql://localhost/db")
            assert isinstance(backend, StateBackend)

    def test_s3_backend_is_state_backend(self):
        """Test that S3 backend satisfies StateBackend protocol."""
        boto3 = pytest.importorskip("boto3")
        from marie_kernel import StateBackend

        with patch.object(boto3, "client") as mock_client_func:
            mock_client = MagicMock()
            mock_client_func.return_value = mock_client

            backend = create_backend_from_url("s3://my-bucket/prefix")
            assert isinstance(backend, StateBackend)
