import os
import subprocess
from pathlib import Path

import pytest

from marie.kb.code.zoekt import Zoekt, search_query


def test_query_is_a_single_scoped_atom_even_for_query_language_input():
    query = search_query('a' * 64, '") or repo:.*', 'literal')
    assert query.startswith('repo:^' + 'a' * 64 + '$ case:yes content:')
    assert '\\"' in query
    with pytest.raises(ValueError):
        search_query('../other', 'anything', 'literal')
    with pytest.raises(ValueError):
        search_query('a' * 64, 'anything', 'unknown')


@pytest.fixture
async def zoekt(tmp_path):
    binaries = os.environ.get('MARIE_ZOEKT_BIN')
    if not binaries:
        pytest.skip('Requires pinned Zoekt binaries')
    import socket

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    served = tmp_path / 'served'
    served.mkdir()
    process = subprocess.Popen(
        [
            str(Path(binaries) / 'zoekt-webserver'),
            '-rpc',
            '-html=false',
            '-listen',
            f'127.0.0.1:{port}',
            '-index',
            str(served),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = Zoekt(
        tmp_path, f'http://127.0.0.1:{port}', str(Path(binaries) / 'zoekt-index')
    )
    try:
        import asyncio

        import httpx

        for _ in range(100):
            try:
                await client.repositories('a' * 64)
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                await asyncio.sleep(0.05)
        yield client
    finally:
        process.terminate()
        process.wait(timeout=10)
        await client.close()


async def test_real_service_build_coverage_scope_and_failed_query(zoekt, tmp_path):
    for handle, content in [
        ('a' * 64, 'alpha secret\nsecond é\n'),
        ('b' * 64, 'beta secret\n'),
    ]:
        source = tmp_path / handle / 'files'
        source.mkdir(parents=True)
        (source / 'same.py').write_text(content)
        await zoekt.build(handle, source, 'main', 'c' * 40, 1)
        assert await zoekt.usable(handle, 1)
    result = await zoekt.search('a' * 64, 'secret', 'literal', 10)
    assert [file['Repository'] for file in result['Files']] == ['a' * 64]
    assert not (await zoekt.search('a' * 64, '") or repo:.*', 'literal', 10)).get(
        'Files'
    )
    assert (await zoekt.search('a' * 64, 's.*t', 'regex', 10))['Files']
    assert (await zoekt.search('a' * 64, 'same.py', 'path', 10))['Files']
    await zoekt.remove('a' * 64)
    assert await zoekt.usable('b' * 64, 1)


async def test_coverage_wait_can_outlast_the_old_fixed_poll_count(
    zoekt, tmp_path, monkeypatch
):
    import asyncio

    files = tmp_path / ('e' * 64) / 'files'
    files.mkdir(parents=True)
    (files / 'one.py').write_text('ready\n')
    usable = zoekt.usable
    sleep = asyncio.sleep
    calls = 0

    async def delayed(*args):
        nonlocal calls
        calls += 1
        return calls > 120 and await usable(*args)

    async def yield_only(seconds):
        await sleep(0)

    monkeypatch.setattr(zoekt, 'usable', delayed)
    monkeypatch.setattr(asyncio, 'sleep', yield_only)
    await zoekt.build('e' * 64, files, 'main', 'c' * 40, 1)
    assert calls > 120 and zoekt.coverage_timeout_seconds > 60
