"""Provider inventory and checkout preparation on the native repository worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator
from urllib.parse import quote, urlparse
from uuid import UUID

import httpx

from marie.kb.item_state import StaleIndexingRun, scheduler_execution
from marie.secret_store.resolver import CredentialResolver


class GitProviderRequestError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(
            f'Git provider request failed ({status_code}); check Integrations'
        )


def clean_url(value: str) -> str:
    url = urlparse(value)
    if (
        url.scheme not in ('http', 'https')
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            'Repository URL must be HTTP(S) without credentials, query or fragment'
        )
    return value.rstrip('/')


def normalize_repository(
    connection: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    gitlab = connection['provider'] == 'gitlab'
    remote = clean_url(row['http_url_to_repo' if gitlab else 'clone_url'])
    instance = clean_url(connection['provider_url'])
    if (
        urlparse(remote).netloc != urlparse(instance).netloc
        or urlparse(remote).scheme != urlparse(instance).scheme
        or not urlparse(remote).path.startswith(
            urlparse(instance).path.rstrip('/') + '/'
        )
    ):
        raise ValueError('Repository clone URL does not belong to its connection')
    full_name = row['path_with_namespace' if gitlab else 'full_name']
    if not isinstance(full_name, str) or '/' not in full_name or not row.get('id'):
        raise ValueError('Provider returned an invalid repository identity')
    return {
        'id': str(row['id']),
        'name': row.get('name') or full_name.rsplit('/', 1)[-1],
        'fullName': full_name,
        'cloneUrl': remote,
        'defaultBranch': row.get('default_branch') or 'main',
        'private': row.get('visibility') != 'public'
        if gitlab
        else bool(row.get('private', True)),
        'description': (row.get('description') or '')[:2000],
        'language': row.get('language'),
        'archived': bool(row.get('archived', False)),
    }


async def provider_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    token: str,
    *,
    provider: str,
    allow_missing: bool = False,
    **kwargs: Any,
) -> Any:
    headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
    if provider == 'gitea':
        headers['Authorization'] = f'token {token}'
    try:
        response = await client.request(method, url, headers=headers, **kwargs)
    except httpx.HTTPError:
        raise RuntimeError(
            'Git provider is unreachable; check Integrations and worker connectivity'
        ) from None
    if response.status_code == 404 and allow_missing:
        return None
    if not response.is_success:
        raise GitProviderRequestError(response.status_code)
    return response.json()


def api_base(connection: dict[str, Any]) -> str:
    base = clean_url(connection['provider_url'])
    if connection['provider'] == 'github':
        return (
            'https://api.github.com'
            if base == 'https://github.com'
            else base + '/api/v3'
        )
    return base + ('/api/v4' if connection['provider'] == 'gitlab' else '/api/v1')


async def discover_repositories(
    client: httpx.AsyncClient, connection: dict[str, Any], token: str
) -> list[dict[str, Any]]:
    provider = connection['provider']
    endpoint = api_base(connection) + (
        '/projects' if provider == 'gitlab' else '/user/repos'
    )
    found: dict[str, dict[str, Any]] = {}
    for page in range(1, 1001):
        params = {'page': page, 'per_page': 100, 'limit': 100}
        if provider == 'gitlab':
            params['membership'] = 'true'
        rows = await provider_request(
            client, 'GET', endpoint, token, provider=provider, params=params
        )
        if not isinstance(rows, list):
            raise ValueError('Git provider returned an invalid inventory page')
        page_ids = {str(row.get('id')) for row in rows}
        if rows and page_ids.issubset(found):
            raise RuntimeError(
                'Git provider repeated an inventory page; completeness could not be verified'
            )
        for row in rows:
            repo = normalize_repository(connection, row)
            found[repo['id']] = repo
        if not rows:
            return sorted(found.values(), key=lambda repo: repo['fullName'].lower())
    raise RuntimeError(
        'Repository inventory exceeded 100000 entries; narrow the host account permissions'
    )


async def verify_registered_repository(
    client: httpx.AsyncClient,
    connection: dict[str, Any],
    registration: dict[str, Any],
    token: str,
    *,
    require_write: bool = False,
) -> None:
    remote_id = registration.get('remote_repository_id')
    if not remote_id:
        raise PermissionError('Repository has no verified provider identity')
    if connection['provider'] == 'gitlab':
        endpoint = api_base(connection) + '/projects/' + quote(str(remote_id), safe='')
    else:
        endpoint = (
            api_base(connection)
            + '/repos/'
            + quote(registration['remote_owner'], safe='')
            + '/'
            + quote(registration['remote_repo_name'], safe='')
        )
    raw = await provider_request(
        client, 'GET', endpoint, token, provider=connection['provider']
    )
    live = normalize_repository(connection, raw)
    if live['id'] != str(remote_id) or live['cloneUrl'] != registration['remote_url']:
        raise PermissionError(
            'Repository identity changed; rediscover and review the registered source'
        )
    if require_write and raw.get('permissions', {}).get('push') is not True:
        raise PermissionError(
            'System Git account requires write access to this repository'
        )
    if (registration.get('metadata') or {}).get('remotePrivate') is True and not live[
        'private'
    ]:
        raise PermissionError(
            'Private repository visibility changed; review its source access'
        )


@contextmanager
def git_environment(username: str, token: str) -> Iterator[dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix='marie-git-auth-') as directory:
        script = Path(directory) / 'askpass'
        script.write_text(
            '#!/bin/sh\ncase "$1" in *Username*) printf "%s\\n" "$MARIE_GIT_USER";; *) printf "%s\\n" "$MARIE_GIT_TOKEN";; esac\n'
        )
        script.chmod(0o700)
        yield {
            **os.environ,
            'GIT_ASKPASS': str(script),
            'GIT_TERMINAL_PROMPT': '0',
            'MARIE_GIT_USER': username,
            'MARIE_GIT_TOKEN': token,
            'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_CONFIG_GLOBAL': '/dev/null',
        }


def run_git(arguments: list[str], *, env: dict[str, str], timeout: int = 300) -> bytes:
    try:
        result = subprocess.run(
            [
                'git',
                '-c',
                'credential.helper=',
                '-c',
                'http.followRedirects=false',
                *arguments,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError('Git operation timed out') from None
    if result.returncode:
        raise RuntimeError(
            'Git operation failed; check repository permissions, branch and worker connectivity'
        )
    return result.stdout


def prepare_checkout(
    remote: str,
    destination: Path,
    *,
    allowed_roots: list[Path],
    env: dict[str, str],
    allow_local: bool = False,
    branch: str | None = None,
    checkout: bool = True,
) -> None:
    if not allow_local:
        clean_url(remote)
    destination = destination.resolve()
    if not any(
        destination.is_relative_to(root.resolve()) and destination != root.resolve()
        for root in allowed_roots
    ):
        raise PermissionError('Repository destination is outside the managed roots')
    if destination.exists():
        try:
            current = (
                run_git(
                    ['-C', str(destination), 'remote', 'get-url', 'origin'], env=env
                )
                .decode()
                .strip()
            )
        except RuntimeError:
            raise ValueError(
                'Repository destination already exists and is not a managed checkout'
            ) from None
        if current != remote:
            raise ValueError(
                'Repository destination already exists with a different origin'
            )
        if branch:
            checked_out = (
                run_git(
                    ['-C', str(destination), 'symbolic-ref', '--short', 'HEAD'], env=env
                )
                .decode()
                .strip()
            )
            if checked_out != branch:
                raise ValueError(
                    'Existing checkout uses a different branch; review workspace topology'
                )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix='.prepare-', dir=destination.parent
    ) as temporary:
        temporary_checkout = Path(temporary) / 'checkout'
        if branch and (branch.startswith('-') or '\x00' in branch):
            raise ValueError('Invalid repository branch')
        run_git(
            [
                'clone',
                *([] if checkout else ['--no-checkout']),
                *(['--branch', branch] if branch else []),
                '--',
                remote,
                str(temporary_checkout),
            ],
            env=env,
        )
        temporary_checkout.rename(destination)


async def import_branch(
    client: httpx.AsyncClient,
    *,
    source: dict[str, Any],
    destination: dict[str, Any],
    source_token: str,
    destination_token: str,
    remote: str,
    branch: str,
    owner: str,
    name: str,
    registration_id: str,
    expected_destination_id: str | None = None,
    record_destination: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    authorize_write: Callable[[], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    if (
        destination['provider'] != 'gitea'
        or not re.fullmatch(r'[A-Za-z0-9_.-]+', owner)
        or not re.fullmatch(r'[A-Za-z0-9_.-]+', name)
    ):
        raise ValueError('Import requires a valid System Git destination')
    if branch.startswith('-') or '\x00' in branch or not branch:
        raise ValueError('Import requires a valid source branch')
    base = api_base(destination)
    endpoint = f'{base}/repos/{quote(owner, safe="")}/{quote(name, safe="")}'
    marker = f'Marie repository import {registration_id}'
    existing = await provider_request(
        client, 'GET', endpoint, destination_token, provider='gitea', allow_missing=True
    )
    if existing and existing.get('private') is not True:
        raise ValueError('System Git import destination must remain private')
    if existing and (
        expected_destination_id is None
        or str(existing.get('id')) != expected_destination_id
    ):
        raise ValueError(
            'System Git destination belongs to a different repository; choose a new name'
        )
    if not existing:
        if expected_destination_id:
            raise ValueError(
                'Recorded System Git destination no longer exists; review the import'
            )
        if authorize_write:
            await authorize_write()
        create = (
            f'{base}/user/repos'
            if owner == destination['provider_username']
            else f'{base}/orgs/{quote(owner, safe="")}/repos'
        )
        existing = await provider_request(
            client,
            'POST',
            create,
            destination_token,
            provider='gitea',
            json={
                'name': name,
                'description': marker,
                'private': True,
                'auto_init': False,
                'default_branch': branch,
            },
        )
    target = normalize_repository(destination, existing)
    if existing.get('private') is not True:
        raise ValueError('System Git import destination must remain private')
    if record_destination:
        await record_destination(target)
    source_user = (
        'oauth2'
        if source['provider'] == 'gitlab'
        else source.get('provider_username') or 'x-access-token'
    )
    with tempfile.TemporaryDirectory(prefix='marie-repository-import-') as directory:
        checkout = Path(directory) / 'source.git'
        with git_environment(source_user, source_token) as env:
            await asyncio.to_thread(
                run_git,
                [
                    'clone',
                    '--bare',
                    '--single-branch',
                    '--branch',
                    branch,
                    '--',
                    clean_url(remote),
                    str(checkout),
                ],
                env=env,
            )
            commit = (
                (
                    await asyncio.to_thread(
                        run_git,
                        ['--git-dir', str(checkout), 'rev-parse', 'HEAD'],
                        env=env,
                    )
                )
                .decode()
                .strip()
            )
        with git_environment(
            destination['provider_username'], destination_token
        ) as env:
            advertised = await asyncio.to_thread(
                run_git, ['ls-remote', '--', target['cloneUrl']], env=env
            )
            refs = [line.split() for line in advertised.decode().splitlines()]
            if any(sha != commit for sha, _ref in refs):
                raise ValueError(
                    'System Git import destination contains different history; review it before retrying'
                )
            if authorize_write:
                await authorize_write()
            latest = await provider_request(
                client, 'GET', endpoint, destination_token, provider='gitea'
            )
            if (
                str(latest.get('id')) != target['id']
                or latest.get('private') is not True
            ):
                raise ValueError(
                    'System Git destination identity or private visibility changed before push'
                )
            await asyncio.to_thread(
                run_git,
                [
                    '--git-dir',
                    str(checkout),
                    'push',
                    '--',
                    target['cloneUrl'],
                    f'HEAD:refs/heads/{branch}',
                ],
                env=env,
            )
    return {**target, 'commit': commit}


def connection_fingerprint(connection: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                str(connection.get(key))
                for key in (
                    'id',
                    'organization_id',
                    'provider',
                    'provider_url',
                    'secret_ref',
                    'is_enabled',
                    'is_valid',
                    'updated_at',
                )
            ]
        ).encode()
    ).hexdigest()


class RepositoryOperations:
    def __init__(self, pool: Any, roots: list[str]) -> None:
        self.pool = pool
        self.roots = [Path(root) for root in roots]
        self.credentials = CredentialResolver()

    async def current_job(self, db: Any, parameters: dict[str, Any]) -> dict[str, Any]:
        execution = scheduler_execution(parameters)
        row = await db.fetchrow(
            '''SELECT j.data,j.dag_id,d.created_on FROM marie_scheduler.job j
            JOIN marie_scheduler.dag d ON d.id=j.dag_id
            WHERE j.id=$1 AND j.state='active' AND j.run_attempt_id=$2 AND j.run_owner=$3
            AND j.run_lease_expires_at > clock_timestamp() FOR UPDATE OF j''',
            UUID(execution['job_id']),
            UUID(execution['run_attempt_id']),
            execution['run_owner'],
        )
        if (
            row is None
            or row['data'].get('metadata', {}).get('planner') != 'repository_management'
        ):
            raise StaleIndexingRun(
                'Repository operations require a current scheduler execution'
            )
        return dict(row)

    async def connection(
        self,
        db: Any,
        connection_id: str,
        organization_id: str,
        scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = await db.fetchrow(
            '''SELECT * FROM marie_runtime.git_provider_connections
            WHERE id=$1 AND ((organization_id=$2 AND user_id IS NULL) OR (user_id=$3 AND organization_id IS NULL))
            AND is_enabled AND is_valid FOR SHARE''',
            UUID(connection_id),
            UUID(organization_id),
            UUID(scope['user_id']) if scope and scope.get('system_git_url') else None,
        )
        if row is None:
            raise PermissionError(
                'Shared Git connection is unavailable; check Integrations'
            )
        result = dict(row)
        if result['user_id'] and (
            not scope
            or result['connection_key'] != 'default'
            or result['provider'] != 'gitea'
            or clean_url(result['provider_url'] or scope.get('system_git_url', ''))
            != clean_url(scope.get('system_git_url', ''))
        ):
            raise PermissionError(
                'Personal repository operations require current System Git authorization'
            )
        if result['user_id'] and not result['provider_url']:
            result['provider_url'] = scope['system_git_url']
        if not result['user_id'] and not re.fullmatch(
            r'env:(?://)?[A-Za-z_][A-Za-z0-9_]*', result['secret_ref'] or ''
        ):
            raise ValueError(
                'Repository worker requires a shared environment secret reference'
            )
        return result

    def token(self, connection: dict[str, Any]) -> str:
        value = (
            self.credentials.resolve_ref(connection['secret_ref'])
            if connection['secret_ref']
            else connection.get('access_token')
        )
        if not value:
            raise ValueError(
                'Git credential reference is not available to the repository worker'
            )
        return value

    async def verify_registration(
        self,
        client: httpx.AsyncClient,
        connection: dict[str, Any],
        registration: dict[str, Any],
        *,
        require_write: bool = False,
    ) -> None:
        try:
            await verify_registered_repository(
                client,
                connection,
                registration,
                self.token(connection),
                require_write=require_write,
            )
        except (GitProviderRequestError, PermissionError) as error:
            if isinstance(error, PermissionError) or error.status_code in (
                401,
                403,
                404,
            ):
                await self.pool.execute(
                    '''UPDATE marie_runtime.git_repositories SET metadata=COALESCE(metadata,'{}'::jsonb) || '{"providerAccessRevoked":true}'::jsonb
                    WHERE id=$1 AND provider_connection_id=$2''',
                    registration['id'],
                    connection['id'],
                )
            if isinstance(error, GitProviderRequestError) and error.status_code == 401:
                await self.pool.execute(
                    'UPDATE marie_runtime.git_provider_connections SET is_valid=false WHERE id=$1',
                    connection['id'],
                )
            raise

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        async with self.pool.acquire() as db:
            async with db.transaction():
                job = await self.current_job(db, parameters)
                scope = job['data']['metadata']
                connection = (
                    await self.connection(
                        db, scope['connection_id'], scope['organization_id'], scope
                    )
                    if scope.get('connection_id')
                    else None
                )
                fingerprint = connection_fingerprint(connection) if connection else None
        operation = scope.get('repository_operation')
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            if operation == 'discover':
                if not connection or connection['user_id']:
                    raise PermissionError(
                        'Shared discovery requires an organization connection'
                    )
                result = {
                    'repositories': await discover_repositories(
                        client, connection, self.token(connection)
                    )
                }
            elif operation == 'prepare':
                result = await self.prepare(scope, connection)
            elif operation == 'import':
                if not connection:
                    raise PermissionError('Import requires a source connection')
                result = await self.import_repository(
                    client, {**scope, '_execution': parameters}, connection
                )
            else:
                raise ValueError('Unsupported repository operation')
        async with self.pool.acquire() as db:
            async with db.transaction():
                await self.current_job(db, parameters)
                if connection:
                    current = await self.connection(
                        db, scope['connection_id'], scope['organization_id'], scope
                    )
                    if connection_fingerprint(current) != fingerprint:
                        raise PermissionError(
                            'Git connection changed during repository operation; retry the job'
                        )
                if operation == 'discover':
                    await db.execute(
                        '''INSERT INTO marie_runtime.git_repository_discovery
                        (connection_id,dag_id,connection_fingerprint,repositories,requested_at)
                        VALUES($1,$2,$3,$4::jsonb,$5) ON CONFLICT(connection_id,dag_id) DO NOTHING''',
                        connection['id'],
                        job['dag_id'],
                        fingerprint,
                        json.dumps(result['repositories']),
                        job['created_on'],
                    )
                    await db.execute(
                        '''UPDATE marie_runtime.git_provider_connections c SET discovered_dag_id=$2
                        WHERE c.id=$1 AND NOT EXISTS(SELECT 1 FROM marie_runtime.git_repository_discovery prior
                            WHERE prior.connection_id=c.id AND prior.dag_id=c.discovered_dag_id AND prior.requested_at > $3)''',
                        connection['id'],
                        job['dag_id'],
                        job['created_on'],
                    )
                elif operation == 'import':
                    await self.finish_import(db, scope, result, job['dag_id'])
                else:
                    changed = await db.fetchrow(
                        '''UPDATE marie_runtime.git_repositories SET path=$1,git_enabled=true,
                        updated_at=clock_timestamp(),metadata=(COALESCE(metadata,'{}'::jsonb) - 'providerAccessRevoked') || $2::jsonb
                        WHERE id=$3 AND provider_connection_id IS NOT DISTINCT FROM $4 AND organization_id=$5 AND is_enabled
                        AND remote_url=$6 AND (path='' OR path=$1) RETURNING id''',
                        result['path'],
                        json.dumps({'preparationDagId': str(job['dag_id'])}),
                        UUID(scope['repository_id']),
                        connection['id'] if connection else None,
                        UUID(scope['organization_id']),
                        result['remoteUrl'],
                    )
                    if changed is None:
                        raise PermissionError(
                            'Repository registration changed during preparation'
                        )
        return (
            {
                'operation': operation,
                'dagId': str(job['dag_id']),
                'count': len(result['repositories']),
            }
            if operation == 'discover'
            else result
        )

    async def prepare(
        self, scope: dict[str, Any], connection: dict[str, Any] | None
    ) -> dict[str, Any]:
        async with self.pool.acquire() as db:
            row = await db.fetchrow(
                '''SELECT * FROM marie_runtime.git_repositories WHERE id=$1
                AND organization_id=$2 AND provider_connection_id IS NOT DISTINCT FROM $3 AND is_enabled AND visibility='organization' ''',
                UUID(scope['repository_id']),
                UUID(scope['organization_id']),
                connection['id'] if connection else None,
            )
        if (
            row is None
            or connection
            and connection['user_id']
            and row['asset_type'] == 'code'
        ):
            raise PermissionError('Repository registration is unavailable')
        if (row['metadata'] or {}).get('upstreamImport') and not (
            row['metadata'] or {}
        ).get('importDagId'):
            raise PermissionError(
                'An import can only be completed by its import workflow'
            )
        if not self.roots:
            raise ValueError('Configure repository_roots before preparing repositories')
        remote = clean_url(row['remote_url'])
        if (
            connection
            and urlparse(remote).netloc != urlparse(connection['provider_url']).netloc
        ):
            raise PermissionError(
                'Registered repository no longer belongs to its Git connection'
            )
        destination = (
            self.roots[0] / 'managed' / scope['organization_id'] / str(row['id'])
        )
        if row['path'] and Path(row['path']).resolve() != destination.resolve():
            raise ValueError(
                'Registered checkout has a different path; review it in workspace topology'
            )
        if connection:
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                await self.verify_registration(
                    client,
                    connection,
                    dict(row),
                    require_write=row['asset_type'] != 'code',
                )
        auth = (
            git_environment(
                'oauth2'
                if connection['provider'] == 'gitlab'
                else connection['provider_username'] or 'x-access-token',
                self.token(connection),
            )
            if connection
            else nullcontext(
                {
                    **os.environ,
                    'GIT_TERMINAL_PROMPT': '0',
                    'GIT_ASKPASS': '/bin/false',
                    'GIT_CONFIG_GLOBAL': '/dev/null',
                }
            )
        )
        with auth as env:
            await asyncio.to_thread(
                prepare_checkout,
                remote,
                destination,
                allowed_roots=self.roots,
                env=env,
                branch=row['git_branch'] if row['asset_type'] != 'code' else None,
                checkout=row['asset_type'] != 'code',
            )
        return {
            'repositoryId': str(row['id']),
            'path': str(destination),
            'remoteUrl': remote,
        }

    async def import_destination(
        self, db: Any, scope: dict[str, Any]
    ) -> dict[str, Any]:
        row = await db.fetchrow(
            '''SELECT * FROM marie_runtime.git_provider_connections WHERE id=$1
            AND user_id=$2 AND organization_id IS NULL AND connection_key='default'
            AND is_enabled AND is_valid FOR SHARE''',
            UUID(scope['destination_connection_id']),
            UUID(scope['user_id']),
        )
        if (
            row is None
            or row['provider'] != 'gitea'
            or clean_url(row['provider_url'] or scope['system_git_url'])
            != clean_url(scope['system_git_url'])
        ):
            raise PermissionError(
                'System Git authorization changed; review the import in Integrations'
            )
        result = dict(row)
        result['provider_url'] = result['provider_url'] or scope['system_git_url']
        return result

    async def import_repository(
        self, client: httpx.AsyncClient, scope: dict[str, Any], source: dict[str, Any]
    ) -> dict[str, Any]:
        async with self.pool.acquire() as db:
            async with db.transaction():
                destination = await self.import_destination(db, scope)
                registered = await db.fetchrow(
                    '''SELECT * FROM marie_runtime.git_repositories WHERE id=$1
                    AND organization_id=$2 AND provider_connection_id IS NOT DISTINCT FROM $3 AND is_enabled''',
                    UUID(scope['repository_id']),
                    UUID(scope['organization_id']),
                    destination['id'],
                )
        if (
            registered is None
            or registered['metadata'].get('upstreamImport') != scope['import_source']
        ):
            raise PermissionError('Import destination registration changed')
        token = (
            self.credentials.resolve_ref(destination['secret_ref'])
            if destination['secret_ref']
            else destination['access_token']
        )
        if not token:
            raise ValueError('System Git account requires reconnection in Integrations')
        source_info = scope['import_source']
        source_path = urlparse(source_info['cloneUrl']).path
        prefix = urlparse(source['provider_url']).path.rstrip('/') + '/'
        relative = source_path[len(prefix) :].removesuffix('.git').rsplit('/', 1)
        if not source_path.startswith(prefix) or len(relative) != 2:
            raise PermissionError('Import source no longer belongs to its connection')
        await verify_registered_repository(
            client,
            source,
            {
                'remote_repository_id': source_info['remoteId'],
                'remote_url': source_info['cloneUrl'],
                'remote_owner': relative[0],
                'remote_repo_name': relative[1],
            },
            self.token(source),
        )

        async def authorize_write() -> None:
            async with self.pool.acquire() as db:
                async with db.transaction():
                    # Caller keeps the scheduler execution parameters outside the public payload.
                    await self.current_job(db, scope['_execution'])
                    current_source = await self.connection(
                        db, scope['connection_id'], scope['organization_id'], scope
                    )
                    current_destination = await self.import_destination(db, scope)
                    if connection_fingerprint(current_source) != connection_fingerprint(
                        source
                    ) or connection_fingerprint(
                        current_destination
                    ) != connection_fingerprint(destination):
                        raise PermissionError(
                            'Repository authorization changed before import write'
                        )

        async def record_destination(target: dict[str, Any]) -> None:
            await authorize_write()
            async with self.pool.acquire() as db:
                changed = await db.fetchrow(
                    '''UPDATE marie_runtime.git_repositories SET remote_repository_id=$1
                    WHERE id=$2 AND organization_id=$3 AND provider_connection_id=$4
                    AND (remote_repository_id IS NULL OR remote_repository_id=$1) AND is_enabled RETURNING id''',
                    target['id'],
                    registered['id'],
                    UUID(scope['organization_id']),
                    destination['id'],
                )
                if changed is None:
                    raise PermissionError('Import destination identity changed')

        result = await import_branch(
            client,
            source=source,
            destination=destination,
            source_token=self.token(source),
            destination_token=token,
            remote=source_info['cloneUrl'],
            branch=source_info['branch'],
            owner=scope['destination_owner'],
            name=scope['destination_name'],
            registration_id=scope['repository_id'],
            expected_destination_id=registered['remote_repository_id'],
            record_destination=record_destination,
            authorize_write=authorize_write,
        )
        checkout = (
            self.roots[0]
            / 'managed'
            / scope['organization_id']
            / scope['repository_id']
        )
        with git_environment(destination['provider_username'], token) as env:
            await asyncio.to_thread(
                prepare_checkout,
                result['cloneUrl'],
                checkout,
                allowed_roots=self.roots,
                env=env,
            )
        return {
            **result,
            'path': str(checkout),
            'destinationFingerprint': connection_fingerprint(destination),
        }

    async def finish_import(
        self, db: Any, scope: dict[str, Any], result: dict[str, Any], dag_id: UUID
    ) -> None:
        destination = await self.import_destination(db, scope)
        if result['destinationFingerprint'] != connection_fingerprint(destination):
            raise PermissionError('System Git authorization changed during import')
        changed = await db.fetchrow(
            '''UPDATE marie_runtime.git_repositories SET path=$1,git_enabled=true,
            remote_url=$2,remote_repository_id=$3,metadata=metadata || $4::jsonb,updated_at=clock_timestamp()
            WHERE id=$5 AND organization_id=$6 AND provider_connection_id=$7 AND is_enabled
            AND metadata->'upstreamImport'=$8::jsonb RETURNING id''',
            result['path'],
            result['cloneUrl'],
            result['id'],
            json.dumps(
                {'importDagId': str(dag_id), 'importedCommit': result['commit']}
            ),
            UUID(scope['repository_id']),
            UUID(scope['organization_id']),
            destination['id'],
            json.dumps(scope['import_source']),
        )
        if changed is None:
            raise PermissionError('Import registration changed during execution')
