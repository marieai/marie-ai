"""Keep one complete repository result on its existing KB source."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import subprocess
import time
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import httpx

from marie.kb.code.browser import (
    directory_entries,
    directory_members,
    file_entries,
    git_blob_digest,
    git_directory_history,
    git_history,
)
from marie.kb.code.capture import (
    branch_commit,
    capture_repository,
    eligible_path,
    resolve_repository_path,
)
from marie.kb.code.capture import (
    code_lines as _code_lines,
)
from marie.kb.code.navigation import parse_file
from marie.kb.code.query import advanced_search
from marie.kb.code.repository_operations import (
    RepositoryOperations,
    git_environment,
    run_git,
)
from marie.kb.code.zoekt import Zoekt
from marie.kb.item_state import StaleIndexingRun, scheduler_execution
from marie.storage.database.postgres_pool import AsyncPostgresConnectionPool


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
        retained_captures: int = 5,
    ) -> None:
        self.pool, self.zoekt, self.roots, self.mounts = pool, zoekt, roots, mounts
        if not 1 <= retained_captures <= 50:
            raise ValueError('Retained captures must be between 1 and 50')
        self.retained_captures = retained_captures

    async def source(
        self, connection: Any, scope: dict[str, Any], *, lock: bool = False
    ) -> dict[str, Any]:
        row = await connection.fetchrow(
            '''SELECT s.*,k.tenant_id,k.workspace_id,k.name AS index_name,t.organization_id,r.path,r.name AS repository_name,r.provider_connection_id,to_jsonb(r) AS repository_registration
               FROM marie_scheduler.kb_source s
               JOIN marie_scheduler.kb_index k ON k.id=s.index_id
               JOIN marie_runtime.tenant t ON t.id=k.tenant_id
               JOIN marie_runtime.git_repositories r ON r.id=(s.dataset_inputs->>'repositoryId')::uuid
               WHERE s.id=$1 AND k.id=$2 AND t.id=$3 AND t.organization_id=$4
                 AND t.status='active' AND s.source_type='repository' AND s.status <> 'disabled'
                 AND NOT COALESCE((k.metadata->>'removalPending')::boolean,false)
                 AND NOT COALESCE((s.metadata->>'removalPending')::boolean,false)
                 AND r.is_enabled AND r.git_enabled
                 AND NOT COALESCE(r.metadata @> '{"providerAccessRevoked":true}'::jsonb,false)
                 AND (r.provider_connection_id IS NULL OR EXISTS(SELECT 1 FROM marie_runtime.git_provider_connections c
                    WHERE c.id=r.provider_connection_id AND c.is_enabled AND c.is_valid
                      AND ((c.organization_id=r.organization_id AND c.user_id IS NULL) OR c.user_id::text=r.metadata->>'organizationAvailabilityApprovedBy')
                      AND (c.provider_url IS NULL AND c.user_id IS NOT NULL AND c.connection_key='default'
                        OR starts_with(r.remote_url,rtrim(c.provider_url,'/') || '/'))))
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

    async def git_credentials(
        self, db: Any, source: dict[str, Any], scope: dict[str, Any]
    ) -> AbstractContextManager[dict[str, str] | None]:
        connection_id = source.get('provider_connection_id')
        if not connection_id:
            return nullcontext(None)
        operations = RepositoryOperations(self.pool, self.roots)
        connection = await operations.connection(
            db, str(connection_id), scope['organization_id']
        )
        registration = source['repository_registration']
        repository = resolve_repository_path(source['path'], self.roots, self.mounts)
        origin = (
            (
                await asyncio.to_thread(
                    run_git,
                    ['-C', str(repository), 'remote', 'get-url', '--all', 'origin'],
                    env={**os.environ, 'GIT_CONFIG_GLOBAL': '/dev/null'},
                )
            )
            .decode()
            .strip()
        )
        if origin != registration['remote_url']:
            raise PermissionError(
                'Checkout origin changed; review it in workspace topology before refreshing'
            )
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            await operations.verify_registration(client, connection, registration)
        username = (
            'oauth2'
            if connection['provider'] == 'gitlab'
            else connection['provider_username'] or 'x-access-token'
        )
        return git_environment(username, operations.token(connection))

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
                with await self.git_credentials(connection, source, scope) as git_env:
                    capture_task = asyncio.create_task(
                        asyncio.to_thread(
                            capture_repository,
                            repository,
                            config['branch'],
                            directory / 'files',
                            dirty=config.get('dirty', False),
                            fetch=config.get('fetch', True),
                            git_env=git_env,
                            remote=source['repository_registration'].get('remote_url')
                            if source.get('provider_connection_id')
                            else None,
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
                        'capturedAt': time.strftime(
                            '%Y-%m-%dT%H:%M:%SZ', time.gmtime()
                        ),
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
                revisions = []
                for item in [
                    result,
                    current,
                    *((source['metadata'] or {}).get('codeRevisions') or []),
                ]:
                    if (
                        not item.get('handle')
                        or item.get('configDigest') != fingerprint
                        or any(saved['handle'] == item['handle'] for saved in revisions)
                    ):
                        continue
                    if (
                        item['handle'] != result['handle']
                        and item.get('commit') == result['commit']
                        and item.get('digest') == result['digest']
                    ):
                        continue
                    revisions.append(item)
                revisions = revisions[: self.retained_captures]
                async with connection.transaction():
                    await connection.execute(
                        'SELECT pg_advisory_xact_lock(hashtextextended($1,60602))', key
                    )
                    await self._job(connection, parameters, lock=True)
                    latest = await self.source(connection, scope, lock=True)
                    if latest.get('provider_connection_id'):
                        await RepositoryOperations(self.pool, self.roots).connection(
                            connection,
                            str(latest['provider_connection_id']),
                            scope['organization_id'],
                        )
                    if settings_digest(latest) != fingerprint:
                        raise StaleIndexingRun(
                            'Repository settings changed during refresh'
                        )
                    await connection.execute(
                        '''UPDATE marie_scheduler.kb_source SET metadata=jsonb_set(jsonb_set(COALESCE(metadata,'{}'),'{codeIndex}',$2::jsonb),'{codeRevisions}',$3::jsonb),
                           status='ready',updated_at=clock_timestamp() WHERE id=$1''',
                        UUID(scope['source_id']),
                        json.dumps(result),
                        json.dumps(revisions),
                    )
                selected = not unchanged
                try:
                    with await self.git_credentials(
                        connection, source, scope
                    ) as git_env:
                        observed = await asyncio.to_thread(
                            branch_commit,
                            repository,
                            config['branch'],
                            fetch=config.get('fetch', True),
                            timeout=5,
                            env=git_env,
                            remote=source['repository_registration'].get('remote_url')
                            if source.get('provider_connection_id')
                            else None,
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
                        retained_handles = [item['handle'] for item in revisions]
                        if (
                            current.get('handle')
                            and current['handle'] not in retained_handles
                        ):
                            await self.zoekt.remove(current['handle'])
                        await self._cleanup(key, retained_handles, connection)
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

    async def _cleanup(self, key: str, keep: str | list[str], connection: Any) -> None:
        handles = [keep] if isinstance(keep, str) else keep
        for manifest in self.zoekt.root.glob('*/manifest.json'):
            if (
                manifest.parent.name not in handles
                and json.loads(manifest.read_text()).get('scope') == key
            ):
                await self.zoekt.remove(manifest.parent.name)
        tenant, index, source = map(UUID, json.loads(key))
        await connection.execute(
            'DELETE FROM marie_scheduler.kb_code_navigation WHERE tenant_id=$1 AND index_id=$2 AND source_id=$3 AND NOT (handle=ANY($4::text[]))',
            tenant,
            index,
            source,
            handles,
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
                        "UPDATE marie_scheduler.kb_source SET metadata=metadata-'codeIndex'-'codeRevisions' WHERE id=$1",
                        source['id'],
                    )

            finally:
                await connection.execute(
                    'SELECT pg_advisory_unlock(hashtextextended($1,60601))', key
                )
        return {'removed': True}

    def _manifest(
        self, source: dict[str, Any], scope: dict[str, Any], handle: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current = (source['metadata'] or {}).get('codeIndex') or {}
        if handle and handle != current.get('handle'):
            current = next(
                (
                    item
                    for item in (source['metadata'] or {}).get('codeRevisions', [])
                    if item.get('handle') == handle
                ),
                current,
            )
        if not current or current.get('configDigest') != settings_digest(source):
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
        self,
        scope: dict[str, Any],
        query: str,
        mode: str,
        limit: int = 20,
        *,
        case_sensitive: bool = True,
        branches: Any = None,
    ) -> dict[str, Any]:
        """Search an authorized current or retained capture with bounded scope."""
        if type(case_sensitive) is not bool:
            raise ValueError('Case sensitivity must be a boolean')
        async with self._read(scope) as (connection, source):
            if mode == 'query':
                return await advanced_search(
                    self, connection, scope, source, branches, case_sensitive, limit
                )
            current, manifest = self._manifest(source, scope)
            if not await self.zoekt.usable(current['handle'], current['files']):
                raise ValueError('Repository index is unavailable; refresh this source')
            result = await self.zoekt.search(
                current['handle'], query, mode, limit, case_sensitive=case_sensitive
            )
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
                    matched = base64.b64decode(line['Line']).decode('utf-8')
                    before = base64.b64decode(line.get('Before') or '').decode('utf-8')
                    after = base64.b64decode(line.get('After') or '').decode('utf-8')
                    context = before + matched + after
                    rows.append(
                        {
                            'path': path,
                            'language': file.get('Language') or 'Unknown',
                            'line': number,
                            'preview': matched[:1000],
                            'context': context[:4000],
                            'contextStartLine': max(1, number - before.count('\n')),
                            'contextTruncated': len(context) > 4000,
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

    async def browse(
        self,
        scope: dict[str, Any],
        citation: dict[str, Any],
        mode: str,
        *,
        path: str = '',
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Browse owned captures and Git metadata under the existing source lock."""
        if (
            mode not in ('tree', 'files', 'revisions', 'history', 'directory_history')
            or not 0 <= offset <= 100000
            or not 1 <= limit <= (1000 if mode == 'files' else 200)
            or (mode == 'files' and path)
        ):
            raise ValueError('Invalid captured repository browse request')
        async with self._read(scope) as (_, source):
            try:
                current, manifest = self._manifest(
                    source, scope, citation.get('handle')
                )
            except ValueError:
                return {
                    'outcome': 'unavailable',
                    'message': 'Captured repository is unavailable. Refresh this source.',
                }
            if citation.get('sourceId') != scope['source_id'] or any(
                citation.get(key) != current[key]
                for key in ('handle', 'commit', 'repositoryId', 'branch')
            ):
                return {
                    'outcome': 'changed',
                    'message': 'Repository capture changed. Search again.',
                }
            info = manifest['files'].get(citation['path'])
            if not info or info['sha256'] != citation['sourceDigest']:
                return {
                    'outcome': 'unavailable',
                    'message': 'Captured file is unavailable.',
                }

            def reference(
                member: str, entry: dict[str, Any], capture: dict[str, Any] = current
            ) -> dict[str, Any]:
                return {
                    **citation,
                    **{
                        key: capture[key]
                        for key in ('handle', 'commit', 'repositoryId', 'branch')
                    },
                    'path': member,
                    'sourceDigest': entry['sha256'],
                    'startLine': 1,
                    'endLine': 1,
                }

            ready = {'outcome': 'ready', 'mode': mode}
            if mode == 'files':
                return {
                    **ready,
                    **file_entries(manifest['files'], citation, offset, limit),
                }
            if mode == 'tree':
                listing = directory_entries(manifest['files'], path, offset, limit)
                return {
                    **ready,
                    **listing,
                    'entries': [
                        {
                            **entry,
                            **(
                                {
                                    'citation': reference(
                                        entry['path'], manifest['files'][entry['path']]
                                    )
                                }
                                if entry['kind'] == 'file'
                                else {}
                            ),
                        }
                        for entry in listing['entries']
                    ],
                }
            if mode == 'directory_history':
                members = directory_members(manifest['files'], path)
                repository = resolve_repository_path(
                    source['path'], self.roots, self.mounts
                )
                try:
                    records = await asyncio.to_thread(
                        git_directory_history,
                        repository,
                        current['commit'],
                        members,
                        offset,
                        limit + 1,
                    )
                except (subprocess.SubprocessError, OSError, ValueError):
                    return {
                        'outcome': 'unavailable',
                        'message': 'Git history is unavailable. Captured files remain readable.',
                    }
                return {
                    **ready,
                    'path': path,
                    'commits': [
                        {**record, 'citation': None} for record in records[:limit]
                    ],
                    'nextOffset': offset + limit if len(records) > limit else None,
                }
            captures = [
                current,
                *((source['metadata'] or {}).get('codeRevisions') or []),
            ]
            seen = set()
            revisions = []
            for capture in captures:
                if capture.get('handle') in seen or capture.get(
                    'configDigest'
                ) != settings_digest(source):
                    continue
                seen.add(capture['handle'])
                saved = self.zoekt.directory(capture['handle']) / 'manifest.json'
                if not saved.is_file():
                    continue
                saved_manifest = json.loads(saved.read_text())
                if saved_manifest.get('scope') != scope_key(scope):
                    raise PermissionError('Captured revision scope differs')
                entry = saved_manifest['files'].get(citation['path'])
                revisions.append(
                    {
                        key: capture.get(key)
                        for key in ('handle', 'commit', 'branch', 'capturedAt')
                    }
                )
                revisions[-1]['citation'] = (
                    reference(citation['path'], entry, capture) if entry else None
                )
            if mode == 'revisions':
                return {
                    **ready,
                    'revisions': revisions,
                    'retentionLimit': self.retained_captures,
                }
            repository = resolve_repository_path(
                source['path'], self.roots, self.mounts
            )
            try:
                if mode == 'history':
                    records = await asyncio.to_thread(
                        git_history,
                        repository,
                        current['commit'],
                        citation['path'],
                        offset,
                        limit + 1,
                    )
                    rows = []
                    for record in records[:limit]:
                        candidates = [
                            r['citation']
                            for r in revisions
                            if r['commit'] == record['commit'] and r['citation']
                        ]
                        digest = (
                            await asyncio.to_thread(
                                git_blob_digest,
                                repository,
                                record['commit'],
                                citation['path'],
                            )
                            if candidates
                            else None
                        )
                        rows.append(
                            {
                                **record,
                                'citation': next(
                                    (
                                        r
                                        for r in candidates
                                        if r['sourceDigest'] == digest
                                    ),
                                    None,
                                ),
                            }
                        )
                    return {
                        **ready,
                        'commits': rows,
                        'nextOffset': offset + limit if len(records) > limit else None,
                    }
            except (subprocess.SubprocessError, OSError, ValueError):
                return {
                    'outcome': 'unavailable',
                    'message': 'Git metadata is unavailable for these captured bytes. The source file remains readable.',
                }

    async def open(
        self,
        scope: dict[str, Any],
        citation: dict[str, Any],
        *,
        whole_file: bool = False,
    ) -> dict[str, Any]:
        """Open only matching captured bytes; never fall back to the checkout."""
        async with self._read(scope) as (_, source):
            try:
                current, manifest = self._manifest(
                    source, scope, citation.get('handle')
                )
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
            if whole_file and len(content) > 2 * 1024 * 1024:
                return {
                    'outcome': 'unavailable',
                    'message': 'This file exceeds the 2 MiB preview size limit.',
                }
            text = content.decode('utf-8')
            lines = _code_lines(text)
            start = 1 if whole_file else _code_line_number(citation['startLine'])
            end = len(lines) if whole_file else _code_line_number(citation['endLine'])
            if not 1 <= start <= end <= len(lines):
                raise ValueError('Code range must contain existing lines')
            passage = ''.join(lines[start - 1 : end])
            if not whole_file and len(passage) > 8000:
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
        *,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Read bounded syntax candidates from the authorized current capture."""
        if mode not in ('outline', 'definitions', 'references'):
            raise ValueError('Unknown code navigation mode')
        if (
            not 1 <= limit <= 50
            or not 0 <= offset <= 100000
            or len(name) > 200
            or (mode != 'outline' and not name.strip())
        ):
            raise ValueError('Navigation requires a bounded name and candidate limit')
        opened = await self.open(scope, citation)
        if opened['outcome'] != 'ready':
            return opened
        async with self._read(scope) as (connection, source):
            try:
                current, manifest = self._manifest(
                    source, scope, citation.get('handle')
                )
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
            if mode == 'references':
                return await self._references(
                    current, manifest, citation, name, limit, offset, coverage, anchor
                )
            array = 'refs' if mode == 'references' else 'symbols'
            captured_lines: dict[str, list[str]] = {}
            legacy_omitted = []
            legacy_rows = await connection.fetch(
                f'''SELECT n.path,n.digest,entry.ordinal,entry.value AS record FROM marie_scheduler.kb_code_navigation n
                CROSS JOIN LATERAL jsonb_array_elements(n.{array}) WITH ORDINALITY entry(value,ordinal)
                WHERE n.tenant_id=$1 AND n.index_id=$2 AND n.source_id=$3 AND n.handle=$4
                AND NOT entry.value ? 'readable'
                AND (($5='outline' AND n.path=$6) OR ($5<>'outline' AND (entry.value->>'name'=$7 OR entry.value->>'qualified_name'=$7)))
                AND ($5<>'outline' OR entry.value->>'kind'<>'parameter')''',
                *args,
                mode,
                citation['path'],
                name,
            )
            for row in legacy_rows:
                info = manifest['files'].get(row['path'])
                if not info or info['sha256'] != row['digest']:
                    raise PermissionError(
                        'Navigation artifact differs from captured file'
                    )
                if row['path'] not in captured_lines:
                    content = await asyncio.to_thread(
                        (
                            self.zoekt.directory(current['handle'])
                            / 'files'
                            / row['path']
                        ).read_bytes
                    )
                    if hashlib.sha256(content).hexdigest() != info['sha256']:
                        return {
                            'outcome': 'changed',
                            'message': 'Captured file digest differs. Refresh this source.',
                        }
                    captured_lines[row['path']] = _code_lines(content.decode('utf-8'))
                start = row['record'].get('line') or row['record']['span']['start_line']
                lines = captured_lines[row['path']]
                if not 1 <= start <= len(lines) or len(lines[start - 1]) > 8000:
                    legacy_omitted.append([row['path'], row['ordinal']])

            rows = await connection.fetch(
                f'''SELECT n.path,n.digest,n.status,entry.value AS record FROM marie_scheduler.kb_code_navigation n
                    CROSS JOIN LATERAL jsonb_array_elements(n.{array}) WITH ORDINALITY entry(value,ordinal)
                    WHERE n.tenant_id=$1 AND n.index_id=$2 AND n.source_id=$3 AND n.handle=$4
                      AND (($5='outline' AND n.path=$6) OR
                        ($5<>'outline' AND (entry.value->>'name'=$7 OR entry.value->>'qualified_name'=$7)))
                      AND ($5<>'outline' OR entry.value->>'kind'<>'parameter')
                      AND COALESCE((entry.value->>'readable')::boolean,true)
                      AND NOT $10::jsonb @> jsonb_build_array(jsonb_build_array(n.path,entry.ordinal))
                    ORDER BY n.path,COALESCE((entry.value->'span'->>'start_line')::int,(entry.value->>'line')::int),
                      entry.value->>'name',entry.ordinal LIMIT $8 OFFSET $9''',
                *args,
                mode,
                citation['path'],
                name,
                limit + 1,
                offset,
                json.dumps(legacy_omitted),
            )
            counts = await connection.fetchrow(
                f'''SELECT count(*) AS matched, count(*) FILTER (WHERE entry.value->>'readable'='false') AS omitted, count(*) FILTER (WHERE NOT entry.value ? 'readable') AS legacy FROM marie_scheduler.kb_code_navigation n
                CROSS JOIN LATERAL jsonb_array_elements(n.{array}) entry(value)
                WHERE n.tenant_id=$1 AND n.index_id=$2 AND n.source_id=$3 AND n.handle=$4
                AND (($5='outline' AND n.path=$6) OR ($5<>'outline' AND (entry.value->>'name'=$7 OR entry.value->>'qualified_name'=$7)))
                AND ($5<>'outline' OR entry.value->>'kind'<>'parameter')''',
                *args,
                mode,
                citation['path'],
                name,
            )
            omitted = counts['omitted'] + len(legacy_omitted)
            total = counts['matched'] - omitted
            complete = not omitted and (
                (anchor is not None and anchor['status'] == 'ready')
                if mode == 'outline'
                else not any(
                    coverage[key]
                    for key in ('partial', 'truncated', 'unsupported', 'unavailable')
                )
            )
            consumed = 0
            results = []
            characters = 0
            truncated = len(rows) > limit or omitted > 0
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
                for line in range(start, min(end, len(lines)) + 1):
                    range_characters += len(lines[line - 1])
                    if range_characters > 8000:
                        break
                    bounded_end = line
                if bounded_end < start:
                    consumed += 1
                    truncated = True
                    continue
                candidate = {
                    'name': record['name'],
                    'qualifiedName': record.get('qualified_name', record['name']),
                    'kind': record['kind'],
                    'signature': record.get('signature', ''),
                    'parseStatus': record.get('parse_status', row['status']),
                    'rangeTruncated': bounded_end < end,
                    'context': ''.join(
                        lines[max(0, start - 3) : min(len(lines), start + 2)]
                    )[:2000],
                    'contextStartLine': max(1, start - 2),
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
                consumed += 1
            return {
                'outcome': 'ready',
                'mode': mode,
                'heuristic': True,
                'status': 'ready' if complete else 'partial',
                'language': anchor['language'] if anchor else None,
                'coverage': coverage,
                'total': total,
                'totalExact': complete,
                'matchedTotal': counts['matched'],
                'omitted': omitted,
                'nextOffset': offset + consumed if offset + consumed < total else None,
                'results': results,
                'truncated': truncated,
            }

    async def _references(
        self,
        current: dict[str, Any],
        manifest: dict[str, Any],
        citation: dict[str, Any],
        name: str,
        limit: int,
        offset: int,
        coverage: dict[str, int],
        anchor: Any,
    ) -> dict[str, Any]:
        found = await self.zoekt.references(current['handle'], name)
        matches = set()
        omitted = set()
        for file in found.get('Files') or []:
            if (
                file['Repository'] != current['handle']
                or file['FileName'] not in manifest['files']
            ):
                raise PermissionError(
                    'Reference search returned a file outside the captured source'
                )
            for line in file.get('LineMatches') or []:
                if line.get('FileName'):
                    continue
                key = (file['FileName'], line['LineNumber'])
                text = base64.b64decode(line['Line']).decode('utf-8')
                if len(text) > 8000:
                    omitted.add(key)
                else:
                    matches.add(key)
        ordered = sorted(matches)
        complete = not (
            found.get('FilesSkipped')
            or found.get('ShardsSkipped')
            or (found.get('FileCount', 0) > len(found.get('Files') or []))
            or found.get('MatchCount', 0) >= 10000
        )
        results = []
        cached = {}
        consumed = 0
        characters = 0
        for path, line in ordered[offset : offset + limit]:
            info = manifest['files'][path]
            if path not in cached:
                content = await asyncio.to_thread(
                    (
                        self.zoekt.directory(current['handle']) / 'files' / path
                    ).read_bytes
                )
                if hashlib.sha256(content).hexdigest() != info['sha256']:
                    return {
                        'outcome': 'changed',
                        'message': 'Captured file digest differs. Refresh this source.',
                    }
                cached[path] = _code_lines(content.decode('utf-8'))
            lines = cached[path]
            if not 1 <= line <= len(lines):
                raise ValueError('Reference line differs from the captured file')
            if len(lines[line - 1]) > 8000:
                consumed += 1
                complete = False
                continue
            candidate = {
                'name': name,
                'qualifiedName': name,
                'kind': 'text',
                'signature': lines[line - 1].strip()[:200],
                'parseStatus': 'ready',
                'rangeTruncated': False,
                'context': ''.join(lines[max(0, line - 3) : min(len(lines), line + 2)])[
                    :2000
                ],
                'contextStartLine': max(1, line - 2),
                'citation': {
                    **citation,
                    'path': path,
                    'sourceDigest': info['sha256'],
                    'startLine': line,
                    'endLine': line,
                },
            }
            size = sum(
                len(candidate[key])
                for key in ('name', 'qualifiedName', 'kind', 'signature')
            ) + len(path)
            if characters + size > 8000:
                break
            characters += size
            consumed += 1
            results.append(candidate)
        next_offset = offset + consumed if offset + consumed < len(ordered) else None
        return {
            'outcome': 'ready',
            'mode': 'references',
            'heuristic': True,
            'status': 'ready' if complete else 'partial',
            'language': anchor['language'] if anchor else None,
            'coverage': coverage,
            'results': results,
            'total': len(ordered),
            'matchedTotal': len(ordered) + len(omitted),
            'omitted': len(omitted),
            'totalExact': complete,
            'nextOffset': next_offset,
            'truncated': next_offset is not None or not complete or bool(omitted),
        }
