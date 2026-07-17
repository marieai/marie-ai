"""End-to-end checks driving the plugin exactly as the daemon does: one command over stdio."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from marie_plugins.runtime.testing import StdioPluginTestClient

PLUGIN_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / 'fixtures'


def _plugin() -> StdioPluginTestClient:
    return StdioPluginTestClient([sys.executable, '-m', 'main'], cwd=PLUGIN_DIR)


def test_pdf_extraction_through_the_real_plugin_command(tmp_path):
    with _plugin() as plugin:
        snapshot = plugin.invoke('capabilities')['data']['data']
        assert snapshot['ready'] is True
        assert 'pdf' in {item['canonical_format'] for item in snapshot['formats']}

        frame = plugin.invoke(
            'extract',
            path=str(FIXTURES / 'sample.pdf'),
            output_dir=str(tmp_path),
        )
        assert frame['data']['type'] == 'stream'
        result = frame['data']['data']
        assert result['outcome'] == 'success'
        assert result['schema_version'] == '1.0'
        assert result['result_kind'] == 'semantic_document'

        descriptor = result['artifact']
        assert descriptor['media_type'] == 'text/markdown'
        artifact = Path(descriptor['path'])
        assert artifact.parent == tmp_path
        body = artifact.read_bytes()
        assert len(body) == descriptor['size_bytes']
        assert hashlib.sha256(body).hexdigest() == descriptor['sha256']
        text = body.decode('utf-8')
        assert text.strip()
        assert text not in json.dumps(result)

        provenance = result['provenance']
        assert provenance['provider']
        assert provenance['canonical_format'] == 'pdf'
        assert result['metadata']['page_count'] == 3
        assert len(text.rstrip('\f').split('\f')) == 3


def test_one_plugin_process_serves_many_sessions(tmp_path):
    with _plugin() as plugin:
        first = plugin.invoke(
            'extract',
            path=str(FIXTURES / 'sample.pdf'),
            output_dir=str(tmp_path / 'a'),
        )['data']['data']
        snapshot = plugin.invoke('capabilities')['data']['data']
        second = plugin.invoke(
            'extract',
            path=str(FIXTURES / 'sample.md'),
            output_dir=str(tmp_path / 'b'),
        )['data']['data']

    assert first['outcome'] == 'success'
    assert snapshot['ready'] is True
    assert second['outcome'] == 'success'
    assert Path(first['artifact']['path']).parent == tmp_path / 'a'
    assert Path(second['artifact']['path']).parent == tmp_path / 'b'


def test_missing_input_returns_typed_error_and_process_survives(tmp_path):
    with _plugin() as plugin:
        frame = plugin.invoke(
            'extract',
            path=str(FIXTURES / 'does-not-exist.pdf'),
            output_dir=str(tmp_path),
        )
        assert frame['data']['type'] == 'error'
        error = frame['data']['data']
        assert error['code'] == 'invalid_request'
        assert error['retryable'] is False

        snapshot = plugin.invoke('capabilities')['data']['data']
        assert snapshot['ready'] is True
