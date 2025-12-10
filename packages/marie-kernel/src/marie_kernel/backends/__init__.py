"""
Marie State Kernel backends.

Available backends:
- InMemoryStateBackend: Thread-safe in-memory backend for testing
- PostgresStateBackend: PostgreSQL backend for production use
- S3StateBackend: Amazon S3 backend for distributed/serverless environments
"""

from marie_kernel.backends.memory import InMemoryStateBackend

__all__ = ["InMemoryStateBackend"]

# PostgresStateBackend requires optional 'postgres' dependency
try:
    from marie_kernel.backends.postgres import PostgresStateBackend

    __all__.append("PostgresStateBackend")
except ImportError:
    pass

# S3StateBackend requires optional 's3' dependency
try:
    from marie_kernel.backends.s3 import S3StateBackend

    __all__.append("S3StateBackend")
except ImportError:
    pass
