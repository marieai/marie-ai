"""Scheduler repository backed by the psycopg 3 async connection pool."""

from marie.scheduler.repository.async_job_repository import AsyncJobRepository


class JobRepository(AsyncJobRepository):
    """Public scheduler repository API."""


__all__ = ["JobRepository"]
