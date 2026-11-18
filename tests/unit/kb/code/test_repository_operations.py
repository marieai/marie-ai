import os
import subprocess
from pathlib import Path

import httpx
import pytest


def test_provider_pages_preserve_identity_and_detect_incomplete_scan():
    from marie.kb.code.repository_operations import normalize_repository

    connection = {'provider': 'gitea', 'provider_url': 'https://git.test'}
    repo = normalize_repository(
        connection,
        {
            'id': 42,
            'full_name': 'team/engine',
            'name': 'engine',
            'clone_url': 'https://git.test/team/engine.git',
            'default_branch': 'main',
            'private': True,
        },
    )
    assert repo['id'] == '42'
    assert repo['fullName'] == 'team/engine'
    assert repo['private'] is True
    with pytest.raises(ValueError):
        normalize_repository(
            connection,
            {
                'id': 42,
                'full_name': 'team/engine',
                'clone_url': 'https://secret@evil.test/repo',
            },
        )


@pytest.mark.asyncio
async def test_paginated_discovery_fails_instead_of_returning_partial_results():
    from marie.kb.code.repository_operations import discover_repositories

    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=[
                    {
                        'id': i,
                        'full_name': f'team/r{i}',
                        'name': f'r{i}',
                        'clone_url': f'https://git.test/team/r{i}.git',
                    }
                    for i in range(1, 101)
                ],
            )
        return httpx.Response(503, text='credential-is-not-an-error-message')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match='Git provider request failed \\(503\\)'):
            await discover_repositories(
                client,
                {'provider': 'gitea', 'provider_url': 'https://git.test'},
                'test-token',
            )
    assert len(calls) == 2


def test_git_credentials_stay_out_of_argv_and_origin(tmp_path, monkeypatch):
    from marie.kb.code.repository_operations import git_environment, run_git

    captured = []

    def run(arguments, **kwargs):
        captured.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, b'', b'')

    monkeypatch.setattr(subprocess, 'run', run)
    with git_environment('service', 'sensitive-value') as env:
        run_git(['ls-remote', 'https://git.test/team/engine.git'], env=env)
        assert Path(env['GIT_ASKPASS']).is_file()
        assert 'sensitive-value' not in Path(env['GIT_ASKPASS']).read_text()
    assert not Path(env['GIT_ASKPASS']).exists()
    assert 'sensitive-value' not in repr(captured[0][0])
    assert captured[0][1]['env']['GIT_TERMINAL_PROMPT'] == '0'


def test_code_preparation_retains_git_objects_without_working_tree_files(tmp_path):
    from marie.kb.code.repository_operations import prepare_checkout

    source = tmp_path / 'source'
    source.mkdir()
    subprocess.run(
        ['git', '-C', str(source), 'init', '-b', 'main'],
        check=True,
        capture_output=True,
    )
    (source / 'sample.py').write_text('enabled = True\n')
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(
        [
            'git',
            '-C',
            str(source),
            '-c',
            'user.name=Fixture',
            '-c',
            'user.email=fixture@example.invalid',
            'commit',
            '-m',
            'initial',
        ],
        check=True,
        capture_output=True,
    )
    destination = tmp_path / 'managed' / 'repository'
    prepare_checkout(
        str(source),
        destination,
        allowed_roots=[tmp_path],
        env=os.environ.copy(),
        allow_local=True,
        checkout=False,
    )
    assert not (destination / 'sample.py').exists()
    assert (
        subprocess.check_output(
            ['git', '-C', str(destination), 'show', 'HEAD:sample.py'], text=True
        )
        == 'enabled = True\n'
    )


def test_prepare_is_idempotent_and_refuses_foreign_existing_directory(tmp_path):
    from marie.kb.code.repository_operations import prepare_checkout

    origin = tmp_path / 'origin.git'
    subprocess.run(
        ['git', 'init', '--bare', str(origin)], check=True, capture_output=True
    )
    destination = tmp_path / 'managed' / 'repository'
    prepare_checkout(
        str(origin),
        destination,
        allowed_roots=[tmp_path],
        env=os.environ.copy(),
        allow_local=True,
    )
    prepare_checkout(
        str(origin),
        destination,
        allowed_roots=[tmp_path],
        env=os.environ.copy(),
        allow_local=True,
    )
    assert subprocess.check_output(
        ['git', '-C', str(destination), 'remote', 'get-url', 'origin'], text=True
    ).strip() == str(origin)
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / 'keep').write_text('user data')
    with pytest.raises(ValueError, match='already exists'):
        prepare_checkout(
            str(origin),
            foreign,
            allowed_roots=[tmp_path],
            env=os.environ.copy(),
            allow_local=True,
        )
    assert (foreign / 'keep').read_text() == 'user data'


@pytest.mark.asyncio
async def test_import_refuses_existing_destination_without_its_durable_marker(
    monkeypatch,
):
    from marie.kb.code.repository_operations import import_branch

    async def handler(request):
        assert request.method == 'GET'
        return httpx.Response(
            200,
            json={
                'id': 77,
                'description': 'Marie repository import stable-registration',
                'private': True,
                'clone_url': 'https://system.test/team/engine.git',
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match='different repository'):
            await import_branch(
                client,
                source={
                    'provider': 'gitea',
                    'provider_url': 'https://upstream.test',
                    'provider_username': 'reader',
                },
                destination={
                    'provider': 'gitea',
                    'provider_url': 'https://system.test',
                    'provider_username': 'team',
                },
                source_token='source',
                destination_token='destination',
                remote='https://upstream.test/team/engine.git',
                branch='main',
                owner='team',
                name='engine',
                registration_id='stable-registration',
            )


@pytest.mark.asyncio
async def test_import_copies_real_branch_history_and_retries_without_overwriting(
    tmp_path, monkeypatch
):
    from marie.kb.code import repository_operations as operations

    source = tmp_path / 'source'
    source.mkdir()
    subprocess.run(
        ['git', '-C', str(source), 'init', '-b', 'main'],
        check=True,
        capture_output=True,
    )
    (source / 'settings.py').write_text('enabled = True\n')
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(
        [
            'git',
            '-C',
            str(source),
            '-c',
            'user.name=Fixture',
            '-c',
            'user.email=fixture@example.invalid',
            'commit',
            '-m',
            'initial',
        ],
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True
    ).strip()
    target = tmp_path / 'target.git'
    created = []

    async def handler(request):
        if request.method == 'GET' and not created:
            return httpx.Response(404)
        if request.method == 'POST':
            subprocess.run(
                ['git', 'init', '--bare', str(target)], check=True, capture_output=True
            )
            created.append(True)
        return httpx.Response(
            200,
            json={
                'id': 77,
                'name': 'engine',
                'full_name': 'team/engine',
                'description': 'Marie repository import stable',
                'clone_url': 'https://system.test/team/engine.git',
                'default_branch': 'main',
                'private': True,
            },
        )

    original = operations.run_git

    def local_git(arguments, **kwargs):
        return original(
            [
                str(source)
                if arg == 'https://upstream.test/team/engine.git'
                else str(target)
                if arg == 'https://system.test/team/engine.git'
                else arg
                for arg in arguments
            ],
            **kwargs,
        )

    monkeypatch.setattr(operations, 'run_git', local_git)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        for attempt in range(2):
            result = await operations.import_branch(
                client,
                source={
                    'provider': 'gitea',
                    'provider_url': 'https://upstream.test',
                    'provider_username': 'reader',
                },
                destination={
                    'provider': 'gitea',
                    'provider_url': 'https://system.test',
                    'provider_username': 'team',
                },
                source_token='source-test',
                destination_token='destination-test',
                remote='https://upstream.test/team/engine.git',
                branch='main',
                owner='team',
                name='engine',
                registration_id='stable',
                expected_destination_id='77' if attempt else None,
            )
            assert result['commit'] == commit
    assert len(created) == 1
    assert (
        subprocess.check_output(
            ['git', '--git-dir', str(target), 'show', 'main:settings.py'], text=True
        )
        == 'enabled = True\n'
    )


@pytest.mark.asyncio
async def test_provider_capped_pages_are_not_mistaken_for_a_complete_inventory():
    from marie.kb.code.repository_operations import discover_repositories

    calls = []

    async def handler(request):
        page = int(request.url.params['page'])
        calls.append(page)
        return httpx.Response(
            200,
            json=[
                {
                    'id': i,
                    'full_name': f'team/r{i}',
                    'clone_url': f'https://git.test/team/r{i}.git',
                }
                for i in ({1: [1, 2], 2: [3]}.get(page, []))
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_repositories(
            client,
            {'provider': 'gitea', 'provider_url': 'https://git.test'},
            'test-token',
        )
    assert [row['id'] for row in result] == ['1', '2', '3']
    assert calls == [1, 2, 3]


@pytest.mark.asyncio
async def test_import_will_not_push_private_source_content_into_a_public_destination(
    monkeypatch,
):
    from marie.kb.code import repository_operations as operations

    async def handler(request):
        return httpx.Response(
            200,
            json={
                'id': 77,
                'name': 'engine',
                'full_name': 'team/engine',
                'description': 'Marie repository import stable',
                'clone_url': 'https://system.test/team/engine.git',
                'private': False,
            },
        )

    monkeypatch.setattr(operations, 'run_git', lambda *args, **kwargs: b'')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match='private'):
            await operations.import_branch(
                client,
                source={
                    'provider': 'gitea',
                    'provider_url': 'https://upstream.test',
                    'provider_username': 'reader',
                },
                destination={
                    'provider': 'gitea',
                    'provider_url': 'https://system.test',
                    'provider_username': 'team',
                },
                source_token='source-test',
                destination_token='destination-test',
                remote='https://upstream.test/team/engine.git',
                branch='main',
                owner='team',
                name='engine',
                registration_id='stable',
            )


@pytest.mark.asyncio
async def test_private_refresh_rejects_origin_drift_before_creating_authenticated_fetch(
    tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from marie.kb.code.store import CodeIndexStore

    root = tmp_path / 'checkout'
    root.mkdir()
    subprocess.run(['git', '-C', str(root), 'init'], check=True, capture_output=True)
    subprocess.run(
        [
            'git',
            '-C',
            str(root),
            'remote',
            'add',
            'origin',
            'https://changed.test/team/repo.git',
        ],
        check=True,
    )
    monkeypatch.setenv('GIT_REFRESH_TEST', 'test-token')
    connection_id = uuid4()
    organization_id = str(uuid4())
    connection = {
        'id': connection_id,
        'provider': 'gitea',
        'provider_url': 'https://approved.test',
        'provider_username': 'reader',
        'user_id': None,
        'secret_ref': 'env:GIT_REFRESH_TEST',
    }
    db = type('Db', (), {'fetchrow': AsyncMock(return_value=connection)})()
    store = CodeIndexStore(None, None, [str(tmp_path)])
    source = {
        'provider_connection_id': connection_id,
        'path': str(root),
        'repository_registration': {
            'remote_url': 'https://approved.test/team/repo.git'
        },
    }
    with pytest.raises(PermissionError, match='origin'):
        await store.git_credentials(db, source, {'organization_id': organization_id})
