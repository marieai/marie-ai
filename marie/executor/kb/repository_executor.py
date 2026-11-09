"""CPU-only native repository jobs and interactive exact-code operations."""

import asyncio
from pathlib import Path
from typing import Any, Dict

from docarray import DocList
from docarray.documents import TextDoc

from marie.executor.marie_executor import MarieExecutor
from marie.kb.code.store import CodeIndexStore
from marie.kb.code.zoekt import Zoekt
from marie.runtime import requests
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool


class RepositoryExecutor(MarieExecutor):
    def __init__(self, code_search: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._code_config = code_search
        self._postgres = kwargs.get('storage', {}).get('psql')
        if not self._postgres:
            raise ValueError(
                'RepositoryExecutor requires the existing storage.psql connection'
            )
        self._store: CodeIndexStore | None = None
        self._lock = asyncio.Lock()

    async def _ensure_store(self) -> CodeIndexStore:
        async with self._lock:
            if self._store is None:
                pool = AsyncPostgresConnectionPool()
                await pool.initialize(self._postgres, autocommit=True)
                config = self._code_config
                self._store = CodeIndexStore(
                    pool,
                    Zoekt(
                        Path(config['root']),
                        config['url'],
                        config.get('indexer', 'zoekt-index'),
                        config.get('coverage_timeout_seconds', 90),
                    ),
                    config['repository_roots'],
                    config.get('path_mounts'),
                )
            return self._store

    @requests(on='/refresh_repository')
    async def refresh_repository(
        self, docs: DocList[TextDoc], parameters: dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return await (await self._ensure_store()).refresh(parameters)

    @requests(on='/code_search')
    async def code_search(
        self, docs: DocList[TextDoc], parameters: dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return await (await self._ensure_store()).search(
            parameters,
            parameters.get('query', ''),
            parameters.get('mode', 'literal'),
            int(parameters.get('limit', 20)),
        )

    @requests(on='/open_code')
    async def open_code(
        self, docs: DocList[TextDoc], parameters: dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return await (await self._ensure_store()).open(
            parameters, parameters['citation']
        )

    @requests(on='/delete_code_source')
    async def delete_code_source(
        self, docs: DocList[TextDoc], parameters: dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return await (await self._ensure_store()).remove(parameters)

    async def close(self) -> None:
        if self._store:
            await self._store.zoekt.close()
            await self._store.pool.close()
