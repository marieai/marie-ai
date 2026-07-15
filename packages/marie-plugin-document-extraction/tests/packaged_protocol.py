"""Packaged stdio checks run inside the plugin uv environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from marie_plugins.runtime.testing import StdioPluginTestClient

PLUGIN_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / 'fixtures'


def test_stdio_capabilities_and_file_backed_extract(tmp_path):
    with StdioPluginTestClient(
        [sys.executable, '-m', 'main'],
        cwd=PLUGIN_DIR,
    ) as plugin:
        capabilities = plugin.invoke('capabilities')
        snapshot = capabilities['data']['data']
        assert snapshot['ready'] is True
        assert 'docx' in {item['canonical_format'] for item in snapshot['formats']}

        extracted = plugin.invoke(
            'extract',
            path=str(FIXTURES / 'sample.html'),
            output_dir=str(tmp_path),
        )
        result = extracted['data']['data']
        assert result['outcome'] == 'success'
        artifact = Path(result['artifact']['path'])
        assert artifact.parent == tmp_path
        assert artifact.read_text().strip()
        assert artifact.read_text() not in json.dumps(result)
