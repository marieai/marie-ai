"""Bounded private-service API for the pinned Zoekt build."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

ZOEKT_REVISION = 'd64299b4420d0f0a46082c26d34ee956184558ed'


def search_query(handle: str, query: str, mode: str) -> str:
    """Encode user input as one atom inside an exact, server-owned build scope."""
    if not re.fullmatch('[a-f0-9]{64}', handle):
        raise ValueError('Invalid code index handle')
    if mode not in ('literal', 'regex', 'path'):
        raise ValueError('Unsupported code search mode')
    if not query or len(query) > 1000 or any(ord(c) < 32 for c in query):
        raise ValueError('A supported code search pattern is required')
    pattern = query if mode == 'regex' else re.escape(query)
    atom = 'file' if mode == 'path' else 'content'
    return f'repo:^{handle}$ case:yes {atom}:{json.dumps(pattern, ensure_ascii=False)}'


class Zoekt:
    """Build separately, verify complete service coverage, then allow selection."""

    def __init__(
        self,
        root: Path,
        url: str,
        indexer: str = 'zoekt-index',
        coverage_timeout_seconds: float = 90,
    ) -> None:
        if not 1 <= coverage_timeout_seconds <= 600:
            raise ValueError('Zoekt coverage timeout must be between 1 and 600 seconds')
        self.coverage_timeout_seconds = coverage_timeout_seconds
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.served = self.root / 'served'
        self.served.mkdir(exist_ok=True)
        self.indexer = str(Path(indexer).absolute()) if '/' in indexer else indexer
        self.client = httpx.AsyncClient(base_url=url, timeout=10, trust_env=False)

    def directory(self, handle: str) -> Path:
        search_query(handle, '.', 'regex')
        return self.root / handle

    async def repositories(self, handle: str) -> dict[str, Any]:
        search_query(handle, '.', 'regex')
        response = await self.client.post('/api/list', json={'Q': f'repo:^{handle}$'})
        response.raise_for_status()
        return response.json()['List']

    async def usable(self, handle: str, count: int) -> bool:
        if count == 0:
            return (self.directory(handle) / 'manifest.json').is_file()
        listed = await self.repositories(handle)
        return (
            not listed.get('Crashes')
            and len(listed.get('Repos') or []) == 1
            and listed['Repos'][0]['Repository']['Name'] == handle
            and listed['Stats']['Documents'] == count
        )

    async def build(
        self, handle: str, files: Path, branch: str, commit: str, count: int
    ) -> dict[str, int]:
        directory = self.directory(handle)
        metadata = directory / 'repository.json'
        metadata.write_text(
            json.dumps(
                {'Name': handle, 'Branches': [{'Name': branch, 'Version': commit}]}
            )
        )
        index = directory / 'index'
        index.mkdir()
        if count == 0:
            return {'shards': 0, 'bytes': 0}
        build_task = asyncio.create_task(
            asyncio.to_thread(
                subprocess.run,
                [
                    self.indexer,
                    '-disable_ctags',
                    '-ignore_dirs',
                    '',
                    '-max_trigram_count',
                    '2000000',
                    '-file_limit',
                    '1048576',
                    '-index',
                    str(index),
                    '-meta',
                    str(metadata),
                    str(files),
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
        )
        try:
            await asyncio.shield(build_task)
        except asyncio.CancelledError:
            await build_task
            raise
        shards = list(index.glob(f'{handle}_*.zoekt'))
        if not shards:
            raise RuntimeError('Zoekt returned no complete shards')
        for shard in shards:
            # Each shard rename is atomic. Nothing selects this handle until all
            # shards are loaded and coverage is verified through the service API.
            shutil.copyfile(shard, self.served / f'{shard.name}.tmp')
            (self.served / f'{shard.name}.tmp').replace(self.served / shard.name)
        deadline = time.monotonic() + self.coverage_timeout_seconds
        while time.monotonic() < deadline:
            if await self.usable(handle, count):
                listed = await self.repositories(handle)
                if listed['Repos'][0]['Repository']['Branches'] != [
                    {'Name': branch, 'Version': commit}
                ]:
                    raise RuntimeError(
                        'Zoekt indexed branch/commit differs from the captured tree'
                    )
                return {
                    'shards': len(shards),
                    'bytes': sum(s.stat().st_size for s in shards),
                }
            await asyncio.sleep(0.1)
        raise RuntimeError(
            'Zoekt index coverage is incomplete; previous index retained'
        )

    async def search(
        self, handle: str, query: str, mode: str, limit: int
    ) -> dict[str, Any]:
        if not 1 <= limit <= 50:
            raise ValueError('Code search limit must be between 1 and 50')
        response = await self.client.post(
            '/api/search',
            json={
                'Q': search_query(handle, query, mode),
                'Opts': {
                    'MaxDocDisplayCount': limit + 1,
                    'TotalMaxMatchCount': 1000,
                    'ShardMaxMatchCount': 1000,
                    'MaxWallTime': 2_000_000_000,
                },
            },
        )
        response.raise_for_status()
        result = response.json()['Result']
        if result.get('Crashes'):
            raise RuntimeError('Zoekt search was incomplete')
        return result

    async def remove(self, handle: str) -> None:
        """Retire exactly this owned handle after callers release their read locks."""
        directory = self.directory(handle)
        for path in self.served.glob(f'{handle}_*.zoekt'):
            path.unlink(missing_ok=True)
        if directory.exists():
            await asyncio.to_thread(shutil.rmtree, directory)

    async def close(self) -> None:
        await self.client.aclose()
