"""
Backend Factory - Create state backends from configuration.

Provides a simple factory function to instantiate backends based on
configuration dictionaries or connection strings.
"""

from typing import Any, Dict, Optional, Union

from marie_kernel.backend import StateBackend
from marie_kernel.backends.memory import InMemoryStateBackend


def create_backend(
    backend_type: str = "memory",
    *,
    connection_pool: Optional[Any] = None,
    s3_client: Optional[Any] = None,
    bucket: Optional[str] = None,
    prefix: str = "marie-state",
    config: Optional[Dict[str, Any]] = None,
) -> StateBackend:
    """
    Create a state backend instance.

    Factory function to create backends based on type string. Supports:
    - "memory": In-memory backend (default, for testing)
    - "postgres": PostgreSQL backend (requires connection_pool)
    - "s3": Amazon S3 backend (requires s3_client and bucket)

    Args:
        backend_type: Type of backend ("memory", "postgres", or "s3")
        connection_pool: psycopg ConnectionPool for postgres backend
        s3_client: boto3 S3 client for S3 backend
        bucket: S3 bucket name for S3 backend
        prefix: S3 key prefix for S3 backend (default: "marie-state")
        config: Optional configuration dict (reserved for future use)

    Returns:
        StateBackend implementation instance

    Raises:
        ValueError: If backend_type is unknown or required params missing
        ImportError: If required dependencies not installed

    Example:
        ```python
        # In-memory for testing
        backend = create_backend("memory")

        # PostgreSQL for production
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool("postgresql://user:pass@localhost/marie")
        backend = create_backend("postgres", connection_pool=pool)

        # S3 for serverless/distributed
        import boto3

        s3 = boto3.client("s3")
        backend = create_backend("s3", s3_client=s3, bucket="my-bucket")
        ```
    """
    backend_type = backend_type.lower()

    if backend_type == "memory":
        return InMemoryStateBackend()

    elif backend_type == "postgres":
        if connection_pool is None:
            raise ValueError(
                "connection_pool is required for postgres backend. "
                "Create one with: psycopg_pool.ConnectionPool(conninfo)"
            )
        from marie_kernel.backends.postgres import PostgresStateBackend

        return PostgresStateBackend(connection_pool)

    elif backend_type == "s3":
        if s3_client is None:
            raise ValueError(
                "s3_client is required for S3 backend. "
                "Create one with: boto3.client('s3')"
            )
        if bucket is None:
            raise ValueError("bucket is required for S3 backend.")
        from marie_kernel.backends.s3 import S3StateBackend

        return S3StateBackend(s3_client, bucket, prefix)

    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported types: 'memory', 'postgres', 's3'"
        )


def create_backend_from_url(url: str) -> StateBackend:
    """
    Create a state backend from a URL/connection string.

    Convenience function that parses a URL to determine backend type
    and creates the appropriate backend.

    Args:
        url: Connection URL or special value
            - "memory://" or "mem://" -> InMemoryStateBackend
            - "postgresql://..." or "postgres://..." -> PostgresStateBackend
            - "s3://bucket/prefix" -> S3StateBackend

    Returns:
        StateBackend implementation instance

    Raises:
        ValueError: If URL scheme is unknown
        ImportError: If required dependencies not installed

    Example:
        ```python
        # In-memory
        backend = create_backend_from_url("memory://")

        # PostgreSQL
        backend = create_backend_from_url("postgresql://user:pass@localhost:5432/marie")

        # S3
        backend = create_backend_from_url("s3://my-bucket/marie-state")
        ```
    """
    url_lower = url.lower()

    if url_lower.startswith(("memory://", "mem://")):
        return InMemoryStateBackend()

    elif url_lower.startswith(("postgresql://", "postgres://")):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError:
            raise ImportError(
                "psycopg is required for PostgreSQL backend. "
                "Install with: pip install marie-kernel[postgres]"
            )
        pool = ConnectionPool(url)
        from marie_kernel.backends.postgres import PostgresStateBackend

        return PostgresStateBackend(pool)

    elif url_lower.startswith("s3://"):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 backend. "
                "Install with: pip install marie-kernel[s3]"
            )
        # Parse s3://bucket/prefix format
        path = url[5:]  # Remove "s3://"
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else "marie-state"

        s3_client = boto3.client("s3")
        from marie_kernel.backends.s3 import S3StateBackend

        return S3StateBackend(s3_client, bucket, prefix)

    else:
        raise ValueError(
            f"Unknown URL scheme in: {url}. "
            f"Supported schemes: 'memory://', 'postgresql://', 's3://'"
        )
