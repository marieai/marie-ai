import base64
import hashlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

from marie.kb.code.store import CodeIndexStore


@pytest.fixture
def captured_file(tmp_path):
    text = ''.join(f'line {line}\n' for line in range(1, 526))
    path = tmp_path / 'files' / 'store.py'
    path.parent.mkdir()
    path.write_text(text)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    store = CodeIndexStore(None, SimpleNamespace(directory=lambda _: tmp_path), [])

    @asynccontextmanager
    async def read(scope: dict[str, Any]) -> AsyncIterator[tuple[None, dict]]:
        yield None, {}

    store._read = read
    store._manifest = lambda *_: (
        {
            'handle': 'capture',
            'commit': 'b' * 40,
            'branch': 'main',
            'repositoryId': 'repository',
            'files': 1,
        },
        {'files': {'store.py': {'sha256': digest}}},
    )
    citation = {
        'handle': 'capture',
        'commit': 'b' * 40,
        'branch': 'main',
        'repositoryId': 'repository',
        'sourceId': 'source',
        'path': 'store.py',
        'sourceDigest': digest,
        'startLine': 33,
        'endLine': 33,
    }
    return store, citation, text


async def test_browser_reads_whole_captured_file_from_single_match(captured_file):
    store, citation, text = captured_file
    result = await store.open({'source_id': 'source'}, citation, whole_file=True)
    assert result['passage'] == text
    assert result['code']['startLine'] == 1
    assert result['code']['endLine'] == 525


async def test_short_excerpt_is_not_restricted_to_one_hundred_lines(captured_file):
    store, citation, _ = captured_file
    result = await store.open(
        {'source_id': 'source'}, {**citation, 'startLine': 1, 'endLine': 250}
    )
    assert result['code']['endLine'] == 250
    assert 'line 250\n' in result['passage']


async def test_whole_file_still_checks_the_saved_digest(captured_file):
    store, citation, _ = captured_file
    result = await store.open(
        {'source_id': 'source'}, {**citation, 'sourceDigest': 'f' * 64}, whole_file=True
    )
    assert result['outcome'] != 'ready'


async def test_search_returns_surrounding_context_with_the_original_match(
    captured_file,
):
    store, _, _ = captured_file

    async def usable(*args):
        return True

    async def search(*args, **kwargs):
        return {
            'Files': [
                {
                    'Repository': 'capture',
                    'FileName': 'store.py',
                    'LineMatches': [
                        {
                            'LineNumber': 33,
                            'Line': base64.b64encode(b'def target():\n').decode(),
                            'Before': base64.b64encode(
                                b'# context 31\n# context 32\n'
                            ).decode(),
                            'After': base64.b64encode(b'    return 1\n').decode(),
                        }
                    ],
                }
            ]
        }

    store.zoekt.usable = usable
    store.zoekt.search = search
    row = (await store.search({'source_id': 'source'}, 'target', 'literal'))['results'][
        0
    ]
    assert row['contextStartLine'] == 31
    assert '# context 32\ndef target():\n    return 1' in row['context']
    assert row['citation']['startLine'] == 33
