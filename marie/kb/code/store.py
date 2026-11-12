"""Keep one complete repository result on its existing KB source."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from marie.kb.code.capture import (
    branch_commit,
    capture_repository,
    eligible_path,
    resolve_repository_path,
)
from marie.kb.code.navigation import parse_file
from marie.kb.code.zoekt import Zoekt
from marie.kb.item_state import StaleIndexingRun, scheduler_execution
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool


def _code_lines(text: str) -> list[str]:
    parts = text.split('\n')
    return [line + '\n' for line in parts[:-1]] + (
        [parts[-1]] if parts[-1] else []
    ) or ['']


def _code_line_number(value: Any) -> int:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError('Code range line numbers must be integers')
    return value


def scope_key(scope: dict[str, Any]) -> str:
    return json.dumps(
        [str(scope[key]) for key in ('tenant_id', 'index_id', 'source_id')]
    )


def settings_digest(source: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps([source['dataset_inputs'], source['path']], sort_keys=True).encode()
    ).hexdigest()


class CodeIndexStore:
    """Reuse source ownership and scheduler leases; Zoekt owns complete shards."""

    def __init__(
        self,
        pool: AsyncPostgresConnectionPool,
        zoekt: Zoekt,
        roots: list[str],
        mounts: dict[str, str] | None = None,
    ) -> None:
        self.pool, self.zoekt, self.roots, self.mounts = pool, zoekt, roots, mounts

    async def source(
        self, connection: Any, scope: dict[str, Any], *, lock: bool = False
    ) -> dict[str, Any]:
        row = await connection.fetchrow(
            '''SELECT s.*,k.tenant_id,k.workspace_id,t.organization_id,r.path,r.name AS repository_name
               FROM marie_scheduler.kb_source s
               JOIN marie_scheduler.kb_index k ON k.id=s.index_id
               JOIN marie_runtime.tenant t ON t.id=k.tenant_id
               JOIN marie_runtime.git_repositories r ON r.id=(s.dataset_inputs->>'repositoryId')::uuid
               WHERE s.id=$1 AND k.id=$2 AND t.id=$3 AND t.organization_id=$4
                 AND t.status='active' AND s.source_type='repository' AND s.status <> 'disabled'
                 AND NOT COALESCE((k.metadata->>'removalPending')::boolean,false)
                 AND NOT COALESCE((s.metadata->>'removalPending')::boolean,false)
                 AND r.is_enabled AND r.git_enabled
                 AND (r.visibility='system' OR (r.visibility='organization' AND r.organization_id=t.organization_id))'''
            + (' FOR UPDATE OF k,s' if lock else ''),
            UUID(scope['source_id']),
            UUID(scope['index_id']),
            UUID(scope['tenant_id']),
            UUID(scope['organization_id']),
        )
        if row is None:
            raise PermissionError('Repository source is unavailable')
        if (
            scope.get('workspace_id')
            and str(row['workspace_id']) != scope['workspace_id']
        ):
            raise PermissionError('Repository source workspace changed')
        return row

    async def _job(
        self, connection: Any, parameters: dict[str, Any], *, lock: bool = False
    ) -> dict[str, Any]:
        execution = scheduler_execution(parameters)
        row = await connection.fetchrow(
            '''SELECT j.data,j.dag_id,d.created_on AS dag_created_on
               FROM marie_scheduler.job j JOIN marie_scheduler.dag d ON d.id=j.dag_id
               WHERE j.id=$1 AND j.state='active' AND j.run_attempt_id=$2 AND j.run_owner=$3
                 AND j.run_lease_expires_at > clock_timestamp()'''
            + (' FOR UPDATE OF j' if lock else ''),
            UUID(execution['job_id']),
            UUID(execution['run_attempt_id']),
            execution['run_owner'],
        )
        if (
            row is None
            or row['data'].get('metadata', {}).get('planner') != 'kb_repository_refresh'
        ):
            raise StaleIndexingRun(
                'Repository refresh requires a current scheduler execution'
            )
        return row

    @asynccontextmanager
    async def _read(
        self, scope: dict[str, Any]
    ) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    'SELECT pg_advisory_xact_lock_shared(hashtextextended($1,60602))',
                    scope_key(scope),
                )
                source = await self.source(connection, scope)
                yield connection, source
                # Read committed recheck also catches revocation during a backend call.
                latest = await self.source(connection, scope)
                if settings_digest(latest) != settings_digest(source):
                    raise PermissionError('Repository source scope changed during read')

    async def refresh(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Capture/build outside selection locks, then select under the job lease."""
        started = time.monotonic()
        handle = ''
        selected = False
        async with self.pool.acquire() as connection:
            job = await self._job(connection, parameters)
            scope = job['data']['metadata']
            key = scope_key(scope)
            if not await connection.fetchval(
                'SELECT pg_try_advisory_lock(hashtextextended($1,60601))', key
            ):
                raise RuntimeError(
                    'Repository refresh already running; scheduler may retry the latest request'
                )
            try:
                source = await self.source(connection, scope)
                config = source['dataset_inputs']
                current = (source['metadata'] or {}).get('codeIndex') or {}
                previous_dag = current.get('checkedDagId') or current.get(
                    'indexedDagId'
                )
                if previous_dag:
                    previous_time = await connection.fetchval(
                        'SELECT created_on FROM marie_scheduler.dag WHERE id=$1',
                        UUID(previous_dag),
                    )
                    if (previous_time or source['updated_at']) > job['dag_created_on']:
                        raise StaleIndexingRun('Repository refresh was superseded')
                repository = resolve_repository_path(
                    source['path'], self.roots, self.mounts
                )
                handle = hashlib.sha256(f'{key}:{uuid4()}'.encode()).hexdigest()
                directory = self.zoekt.directory(handle)
                directory.mkdir()
                (directory / 'manifest.json').write_text(
                    json.dumps({'scope': key, 'files': {}})
                )
                capture_task = asyncio.create_task(
                    asyncio.to_thread(
                        capture_repository,
                        repository,
                        config['branch'],
                        directory / 'files',
                        dirty=config.get('dirty', False),
                        fetch=config.get('fetch', True),
                        include=config.get('include'),
                        exclude=config.get('exclude'),
                    )
                )
                try:
                    capture = await asyncio.shield(capture_task)
                except asyncio.CancelledError:
                    await capture_task
                    raise
                manifest = {
                    'scope': key,
                    'files': capture.files,
                    'commit': capture.commit,
                }
                (directory / 'manifest.json').write_text(json.dumps(manifest))
                fingerprint = settings_digest(source)
                navigation = await asyncio.to_thread(
                    self._parse_capture, directory / 'files', capture.files
                )
                navigation_digest = hashlib.sha256(
                    json.dumps(navigation, sort_keys=True).encode()
                ).hexdigest()
                previous_files = {}
                previous_valid = False
                if current.get('handle'):
                    previous_manifest = (
                        self.zoekt.directory(current['handle']) / 'manifest.json'
                    )
                    try:
                        previous = json.loads(previous_manifest.read_text())
                        previous_valid = previous.get('scope') == key and isinstance(
                            previous.get('files'), dict
                        )
                        if previous_valid:
                            previous_files = previous['files']
                    except (OSError, json.JSONDecodeError):
                        pass
                unchanged = bool(
                    previous_valid
                    and current.get('digest') == capture.digest
                    and current.get('configDigest') == fingerprint
                    and current.get('navigationDigest') == navigation_digest
                    and await self.zoekt.usable(current['handle'], current['files'])
                )
                changed = sum(
                    previous_files.get(p) != value for p, value in capture.files.items()
                )
                deleted = len(previous_files.keys() - capture.files.keys())
                stats = {'shards': 0, 'bytes': 0}
                if not unchanged:
                    stats = await self.zoekt.build(
                        handle,
                        directory / 'files',
                        config['branch'],
                        capture.commit,
                        len(capture.files),
                    )
                    await self._save_navigation(connection, scope, handle, navigation)
                result = (
                    current
                    if unchanged
                    else {
                        'handle': handle,
                        'commit': capture.commit,
                        'branch': config['branch'],
                        'repositoryId': config['repositoryId'],
                        'configDigest': fingerprint,
                        'digest': capture.digest,
                        'files': len(capture.files),
                        'size': capture.size,
                        'omitted': capture.omitted,
                        'indexedDagId': str(job['dag_id']),
                        'navigationDigest': navigation_digest,
                        **stats,
                    }
                )
                result = {
                    **result,
                    'checkedCommit': capture.commit,
                    'checkedDagId': str(job['dag_id']),
                    'skipped': unchanged,
                    'changed': changed,
                    'deleted': deleted,
                    'duration': round(time.monotonic() - started, 4),
                }
                result.pop('pendingCommit', None)
                result.pop('cleanupError', None)
                async with connection.transaction():
                    await connection.execute(
                        'SELECT pg_advisory_xact_lock(hashtextextended($1,60602))', key
                    )
                    await self._job(connection, parameters, lock=True)
                    latest = await self.source(connection, scope, lock=True)
                    if settings_digest(latest) != fingerprint:
                        raise StaleIndexingRun(
                            'Repository settings changed during refresh'
                        )
                    await connection.execute(
                        '''UPDATE marie_scheduler.kb_source SET metadata=jsonb_set(COALESCE(metadata,'{}'),'{codeIndex}',$2::jsonb),
                           status='ready',updated_at=clock_timestamp() WHERE id=$1''',
                        UUID(scope['source_id']),
                        json.dumps(result),
                    )
                selected = not unchanged
                try:
                    observed = await asyncio.to_thread(
                        branch_commit,
                        repository,
                        config['branch'],
                        fetch=config.get('fetch', True),
                        timeout=5,
                    )
                except (subprocess.SubprocessError, OSError, ValueError):
                    observed = capture.commit
                if observed != capture.commit:
                    result['pendingCommit'] = observed
                    await connection.execute(
                        "UPDATE marie_scheduler.kb_source SET metadata=jsonb_set(metadata,'{codeIndex,pendingCommit}',$2::jsonb) WHERE id=$1 AND metadata->'codeIndex'->>'handle'=$3 AND dataset_inputs=$4::jsonb",
                        UUID(scope['source_id']),
                        json.dumps(observed),
                        result['handle'],
                        json.dumps(config),
                    )
                try:
                    # Wait for earlier readers before retiring the superseded build.
                    async with connection.transaction():
                        await connection.execute(
                            'SELECT pg_advisory_xact_lock(hashtextextended($1,60602))',
                            key,
                        )
                        if (
                            current.get('handle')
                            and current['handle'] != result['handle']
                        ):
                            await self.zoekt.remove(current['handle'])
                        await self._cleanup(key, result['handle'], connection)
                except (OSError, json.JSONDecodeError):
                    result['cleanupError'] = (
                        'Owned retired repository files could not be removed; the next refresh retries cleanup.'
                    )
                    await connection.execute(
                        "UPDATE marie_scheduler.kb_source SET metadata=jsonb_set(metadata,'{codeIndex,cleanupError}',$2::jsonb) WHERE id=$1 AND metadata->'codeIndex'->>'handle'=$3",
                        UUID(scope['source_id']),
                        json.dumps(result['cleanupError']),
                        result['handle'],
                    )
                return result
            finally:
                try:
                    if handle and not selected:
                        saved = await connection.fetchval(
                            "SELECT metadata->'codeIndex'->>'handle' FROM marie_scheduler.kb_source WHERE id=$1",
                            UUID(scope['source_id']),
                        )
                        if saved != handle:
                            await self.zoekt.remove(handle)
                            await connection.execute(
                                'DELETE FROM marie_scheduler.kb_code_navigation WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND handle=$4',
                                UUID(scope['tenant_id']),
                                UUID(scope['index_id']),
                                UUID(scope['source_id']),
                                handle,
                            )
                finally:
                    await connection.execute(
                        'SELECT pg_advisory_unlock(hashtextextended($1,60601))', key
                    )

    def _parse_capture(
        self, directory: Path, files: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            {
                'path': path,
                'digest': info['sha256'],
                **parse_file(directory / path, info['sha256']),
            }
            for path, info in sorted(files.items())
        ]

    async def _save_navigation(
        self,
        connection: Any,
        scope: dict[str, Any],
        handle: str,
        records: list[dict[str, Any]],
    ) -> None:
        for start in range(0, len(records), 100):
            await connection.execute(
                '''INSERT INTO marie_scheduler.kb_code_navigation
                   (tenant_id,index_id,source_id,handle,path,digest,language,status,provider_version,symbols,refs,imports)
                   SELECT $1,$2,$3,$4,r.path,r.digest,r.language,r.status,r.provider_version,r.symbols,r.refs,r.imports
                   FROM jsonb_to_recordset($5::jsonb) AS r(path text,digest text,language text,status text,provider_version text,
                     symbols jsonb,refs jsonb,imports jsonb)''',
                UUID(scope['tenant_id']),
                UUID(scope['index_id']),
                UUID(scope['source_id']),
                handle,
                json.dumps(
                    [
                        {**row, 'refs': row['references']}
                        for row in records[start : start + 100]
                    ]
                ),
            )

    async def _cleanup(self, key: str, keep: str, connection: Any) -> None:
        for manifest in self.zoekt.root.glob('*/manifest.json'):
            if (
                manifest.parent.name != keep
                and json.loads(manifest.read_text()).get('scope') == key
            ):
                await self.zoekt.remove(manifest.parent.name)
        tenant, index, source = map(UUID, json.loads(key))
        await connection.execute(
            'DELETE FROM marie_scheduler.kb_code_navigation WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND handle<>$4',
            tenant,
            index,
            source,
            keep,
        )

    async def remove(self, scope: dict[str, Any]) -> dict[str, Any]:
        """Withdraw a source, including removal-pending sources, and its owned shards."""
        async with self.pool.acquire() as connection:
            key = scope_key(scope)
            if not await connection.fetchval(
                'SELECT pg_try_advisory_lock(hashtextextended($1,60601))', key
            ):
                raise RuntimeError(
                    'Repository refresh is still stopping; retry source removal'
                )
            try:
                async with connection.transaction():
                    await connection.execute(
                        'SELECT pg_advisory_xact_lock(hashtextextended($1,60602))',
                        scope_key(scope),
                    )
                    source = await connection.fetchrow(
                        '''SELECT s.id,s.metadata->'codeIndex'->>'handle' AS handle FROM marie_scheduler.kb_source s JOIN marie_scheduler.kb_index k ON k.id=s.index_id
                           JOIN marie_runtime.tenant t ON t.id=k.tenant_id
                           WHERE s.id=$1 AND k.id=$2 AND t.id=$3 AND t.organization_id=$4 AND s.source_type='repository'
                           FOR UPDATE OF k,s''',
                        UUID(scope['source_id']),
                        UUID(scope['index_id']),
                        UUID(scope['tenant_id']),
                        UUID(scope['organization_id']),
                    )
                    if source is None:
                        raise PermissionError('Repository source is unavailable')
                    await connection.execute(
                        "UPDATE marie_scheduler.kb_source SET status='disabled',updated_at=clock_timestamp() WHERE id=$1",
                        source['id'],
                    )
                async with connection.transaction():
                    await connection.execute(
                        'SELECT pg_advisory_xact_lock(hashtextextended($1,60602))',
                        scope_key(scope),
                    )
                    if source['handle']:
                        await self.zoekt.remove(source['handle'])
                    await self._cleanup(scope_key(scope), '', connection)
                    await connection.execute(
                        "UPDATE marie_scheduler.kb_source SET metadata=metadata-'codeIndex' WHERE id=$1",
                        source['id'],
                    )

            finally:
                await connection.execute(
                    'SELECT pg_advisory_unlock(hashtextextended($1,60601))', key
                )
        return {'removed': True}

    def _manifest(
        self, source: dict[str, Any], scope: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current = (source['metadata'] or {}).get('codeIndex') or {}
        if not current or current['configDigest'] != settings_digest(source):
            raise ValueError('Repository index is unavailable; refresh this source')
        path = self.zoekt.directory(current['handle']) / 'manifest.json'
        if not path.is_file():
            raise ValueError('Repository index is unavailable; refresh this source')
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                'Captured repository manifest is unavailable; refresh this source'
            ) from error
        if manifest['scope'] != scope_key(scope):
            raise PermissionError('Repository index scope differs')
        return current, manifest

    async def search(
        self, scope: dict[str, Any], query: str, mode: str, limit: int = 20
    ) -> dict[str, Any]:
        """Search one authorized current tree with a bounded, exact build filter."""
        async with self._read(scope) as (_, source):
            current, manifest = self._manifest(source, scope)
            if not await self.zoekt.usable(current['handle'], current['files']):
                raise ValueError('Repository index is unavailable; refresh this source')
            result = await self.zoekt.search(current['handle'], query, mode, limit)
            rows = []
            for file in (result.get('Files') or [])[:limit]:
                path = file['FileName']
                if (
                    file['Repository'] != current['handle']
                    or path not in manifest['files']
                ):
                    raise PermissionError(
                        'Zoekt returned a file outside the captured scope'
                    )
                for line in (
                    file.get('LineMatches') or [{'LineNumber': 1, 'Line': ''}]
                )[:10]:
                    number = max(1, line['LineNumber'])
                    rows.append(
                        {
                            'path': path,
                            'line': number,
                            'preview': base64.b64decode(line['Line']).decode('utf-8')[
                                :1000
                            ],
                            'citation': {
                                'kind': 'code',
                                'sourceId': scope['source_id'],
                                'handle': current['handle'],
                                'repositoryId': current['repositoryId'],
                                'branch': current['branch'],
                                'commit': current['commit'],
                                'path': path,
                                'sourceDigest': manifest['files'][path]['sha256'],
                                'startLine': number,
                                'endLine': number,
                            },
                        }
                    )
            return {
                'results': rows[:limit],
                'truncated': len(rows) > limit
                or len(result.get('Files') or []) > limit
                or bool(
                    result.get('FilesSkipped') or result.get('MatchCount', 0) > limit
                ),
                'indexed': current,
                'backend': 'zoekt',
            }

    async def open(
        self, scope: dict[str, Any], citation: dict[str, Any]
    ) -> dict[str, Any]:
        """Open only matching captured bytes; never fall back to the checkout."""
        async with self._read(scope) as (_, source):
            try:
                current, manifest = self._manifest(source, scope)
            except ValueError:
                return {
                    'outcome': 'unavailable',
                    'message': 'Captured repository is unavailable. Refresh this source.',
                }
            if citation['sourceId'] != scope['source_id']:
                raise PermissionError('Code citation belongs to another source')
            if any(
                citation.get(k) != current[k]
                for k in ('handle', 'branch', 'commit', 'repositoryId')
            ):
                return {
                    'outcome': 'changed',
                    'message': 'This repository has changed. Search again.',
                }
            path = citation['path']
            info = manifest['files'].get(path)
            if not info or citation['sourceDigest'] != info['sha256']:
                return {
                    'outcome': 'unavailable',
                    'message': 'Captured file is unavailable.',
                }
            original = self.zoekt.directory(current['handle']) / 'files' / path
            if not original.is_file():
                return {
                    'outcome': 'unavailable',
                    'message': 'Captured file is unavailable.',
                }
            try:
                content = await asyncio.to_thread(original.read_bytes)
            except OSError:
                return {
                    'outcome': 'unavailable',
                    'message': 'Captured file is unavailable.',
                }
            if hashlib.sha256(content).hexdigest() != info['sha256']:
                return {
                    'outcome': 'changed',
                    'message': 'Captured file digest differs. Refresh this source.',
                }
            text = content.decode('utf-8')
            lines = _code_lines(text)
            start = _code_line_number(citation['startLine'])
            end = _code_line_number(citation['endLine'])
            if not 1 <= start <= end <= len(lines) or end - start >= 100:
                raise ValueError('Code range must contain at most 100 existing lines')
            passage = ''.join(lines[start - 1 : end])
            if len(passage) > 8000:
                raise ValueError(
                    'Code range exceeds character budget; select fewer lines'
                )
            offset = sum(map(len, lines[: start - 1]))
            return {
                'outcome': 'ready',
                'text': passage,
                'passage': passage,
                'start_offset': offset,
                'end_offset': offset + len(passage),
                'source_digest': info['sha256'],
                'code': {
                    'path': path,
                    'branch': current['branch'],
                    'commit': current['commit'],
                    'startLine': start,
                    'endLine': end,
                    'totalLines': len(lines),
                },
            }

    async def navigate(
        self,
        scope: dict[str, Any],
        citation: dict[str, Any],
        mode: str,
        name: str = '',
        limit: int = 20,
    ) -> dict[str, Any]:
        """Read bounded syntax candidates from the authorized current capture."""
        if mode not in ('outline', 'definitions', 'references'):
            raise ValueError('Unknown code navigation mode')
        if (
            not 1 <= limit <= 50
            or len(name) > 200
            or (mode != 'outline' and not name.strip())
        ):
            raise ValueError('Navigation requires a bounded name and candidate limit')
        opened = await self.open(scope, citation)
        if opened['outcome'] != 'ready':
            return opened
        async with self._read(scope) as (connection, source):
            try:
                current, manifest = self._manifest(source, scope)
            except ValueError:
                return {
                    'outcome': 'unavailable',
                    'message': 'Captured repository is unavailable. Refresh this source.',
                }
            if any(
                citation.get(key) != current[key]
                for key in ('handle', 'repositoryId', 'branch', 'commit')
            ):
                return {
                    'outcome': 'changed',
                    'message': 'This repository has changed. Search again.',
                }
            args = [
                UUID(scope['tenant_id']),
                UUID(scope['index_id']),
                UUID(scope['source_id']),
                current['handle'],
            ]
            coverage_rows = await connection.fetch(
                '''SELECT status,count(*)::int AS count FROM marie_scheduler.kb_code_navigation
                   WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND handle=$4 GROUP BY status''',
                *args,
            )
            coverage = {
                status: 0
                for status in (
                    'ready',
                    'partial',
                    'truncated',
                    'unsupported',
                    'unavailable',
                )
            }
            coverage.update({row['status']: row['count'] for row in coverage_rows})
            coverage['unavailable'] += max(
                0, len(manifest['files']) - sum(coverage.values())
            )
            anchor = await connection.fetchrow(
                '''SELECT status,language FROM marie_scheduler.kb_code_navigation
                   WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND handle=$4 AND path=$5''',
                *args,
                citation['path'],
            )
            array = 'refs' if mode == 'references' else 'symbols'
            rows = await connection.fetch(
                f'''SELECT n.path,n.digest,n.status,entry.value AS record FROM marie_scheduler.kb_code_navigation n
                    CROSS JOIN LATERAL jsonb_array_elements(n.{array}) entry(value)
                    WHERE n.tenant_id=$1 AND n.index_id=$2 AND n.source_id=$3 AND n.handle=$4
                      AND (($5='outline' AND n.path=$6) OR
                        ($5<>'outline' AND (entry.value->>'name'=$7 OR entry.value->>'qualified_name'=$7)))
                    ORDER BY n.path,COALESCE((entry.value->'span'->>'start_line')::int,(entry.value->>'line')::int),
                      entry.value->>'name' LIMIT $8''',
                *args,
                mode,
                citation['path'],
                name,
                limit + 1,
            )
            results = []
            captured_lines: dict[str, list[str]] = {}
            characters = 0
            truncated = len(rows) > limit
            for row in rows[:limit]:
                info = manifest['files'].get(row['path'])
                if not info or info['sha256'] != row['digest']:
                    raise PermissionError(
                        'Navigation artifact differs from captured file'
                    )
                record = row['record']
                start = record.get('line') or record['span']['start_line']
                span = record.get('span') or {}
                end = max(
                    start, span.get('end_line', start) - (span.get('end_col') == 0)
                )
                if row['path'] not in captured_lines:
                    original = (
                        self.zoekt.directory(current['handle']) / 'files' / row['path']
                    )
                    try:
                        content = await asyncio.to_thread(original.read_bytes)
                    except OSError:
                        return {
                            'outcome': 'unavailable',
                            'message': 'Captured file is unavailable.',
                        }
                    if hashlib.sha256(content).hexdigest() != info['sha256']:
                        return {
                            'outcome': 'changed',
                            'message': 'Captured file digest differs. Refresh this source.',
                        }
                    captured_lines[row['path']] = _code_lines(content.decode('utf-8'))
                lines = captured_lines[row['path']]
                bounded_end = start - 1
                range_characters = 0
                for line in range(start, min(end, start + 99, len(lines)) + 1):
                    range_characters += len(lines[line - 1])
                    if range_characters > 8000:
                        break
                    bounded_end = line
                if bounded_end < start:
                    truncated = True
                    continue
                candidate = {
                    'name': record['name'],
                    'qualifiedName': record.get('qualified_name', record['name']),
                    'kind': record['kind'],
                    'signature': record.get('signature', ''),
                    'parseStatus': record.get('parse_status', row['status']),
                    'rangeTruncated': bounded_end < end,
                    'citation': {
                        **citation,
                        'path': row['path'],
                        'sourceDigest': row['digest'],
                        'startLine': start,
                        'endLine': bounded_end,
                    },
                }
                size = sum(
                    len(candidate[key])
                    for key in ('name', 'qualifiedName', 'kind', 'signature')
                ) + len(row['path'])
                if characters + size > 8000:
                    truncated = True
                    break
                characters += size
                results.append(candidate)
            return {
                'outcome': 'ready',
                'mode': mode,
                'heuristic': True,
                'status': anchor['status'] if anchor else 'unavailable',
                'language': anchor['language'] if anchor else None,
                'coverage': coverage,
                'results': results,
                'truncated': truncated,
            }
