"""Compatibility import for the psycopg 3 asynchronous PostgreSQL pool.

The historical module name is retained for existing callers.
"""
from marie.storage.database.postgres_pool import AsyncPostgresPool

__all__ = ["AsyncPostgresPool"]
