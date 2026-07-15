"""Exercise document extraction through EmbeddedPlugins and the real daemon."""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from marie.plugins.embedded import EmbeddedPlugins

REPO_ROOT = Path(__file__).resolve().parents[3]
DAEMON_DIR = REPO_ROOT / 'packages' / 'marie-plugin-daemon'
PLUGIN_DIR = REPO_ROOT / 'packages' / 'marie-plugin-document-extraction'
PACKAGE = 'marie/document-extraction'

EXPECTED_FORMATS = {
    'arduino',
    'bash',
    'c',
    'chatito',
    'clojure',
    'commonlisp',
    'cpp',
    'csharp',
    'csv',
    'd',
    'dart',
    'docx',
    'elisp',
    'elixir',
    'elm',
    'eml',
    'epub',
    'fortran',
    'gleam',
    'go',
    'haskell',
    'hcl',
    'html',
    'java',
    'javascript',
    'kotlin',
    'latex',
    'lua',
    'markdown',
    'matlab',
    'ocaml',
    'ocaml_interface',
    'odp',
    'ods',
    'odt',
    'pdf',
    'php',
    'pony',
    'pptx',
    'properties',
    'python',
    'ql',
    'r',
    'racket',
    'ruby',
    'rust',
    'scala',
    'solidity',
    'swift',
    'typescript',
    'udev',
    'xlsx',
    'zig',
}


def _daemon_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    return f'127.0.0.1:{port}'


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_embedded_plugins_invokes_packaged_document_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if shutil.which('go') is None:
        pytest.skip('Go is required to build marie-plugin-daemon')
    if shutil.which('uv') is None:
        pytest.skip('uv is required to bootstrap the plugin environment')

    daemon_bin = tmp_path / 'marie-plugin-daemon'
    subprocess.run(
        ['go', 'build', '-o', str(daemon_bin), './cmd/server'],
        cwd=DAEMON_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    archive_dir = tmp_path / 'archives'
    subprocess.run(
        ['bash', str(PLUGIN_DIR / 'scripts' / 'package.sh'), str(archive_dir)],
        cwd=PLUGIN_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    archives = list(archive_dir.glob('marie-plugin-document-extraction_*.zip'))
    assert len(archives) == 1

    monkeypatch.delenv('MARIE_PLUGIN_DAEMON_URL', raising=False)
    monkeypatch.setenv('MARIE_PLUGIN_DAEMON_BIN', str(daemon_bin))
    monkeypatch.setenv('MARIE_PLUGIN_STORAGE_ROOT', str(tmp_path / 'storage'))
    monkeypatch.setenv('MARIE_PLUGIN_DAEMON_LOG_LEVEL', 'ERROR')
    monkeypatch.setenv(
        'MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID', 'marie-executor-embedded'
    )
    monkeypatch.setenv(
        'MARIE_PLUGIN_DAEMON_SIGNING_SECRET',
        'marie-executor-embedded-loopback-secret',
    )

    config = [
        {
            'package': PACKAGE,
            'path': str(archives[0]),
            'actions': ['capabilities', 'extract'],
            'timeout_s': 120,
        }
    ]
    with EmbeddedPlugins(
        config,
        executor_identity='document-extraction-integration',
        daemon_addr=_daemon_address(),
    ) as plugins:
        capabilities = plugins.capabilities(PACKAGE)
        formats = {
            item['canonical_format'] for item in capabilities.get('formats', [])
        }
        assert capabilities['ready'] is True
        assert formats == EXPECTED_FORMATS

        output_dir = tmp_path / 'output'
        result = plugins.invoke(
            PACKAGE,
            'extract',
            {
                'path': str(PLUGIN_DIR / 'tests' / 'fixtures' / 'sample.html'),
                'output_dir': str(output_dir),
            },
        )

    assert result['outcome'] == 'success'
    assert result['provenance']['canonical_format'] == 'html'
    artifact = Path(result['artifact']['path'])
    assert artifact.parent == output_dir
    assert 'This is some text.' in artifact.read_text()
