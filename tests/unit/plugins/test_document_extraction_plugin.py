import ast
import importlib.util
import json
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = REPO_ROOT / 'packages' / 'marie-plugin-document-extraction'
RUNTIME_DIR = REPO_ROOT / 'packages' / 'marie-plugin-daemon' / 'python_runtime'
sys.path.insert(0, str(RUNTIME_DIR))
sys.path.insert(0, str(PLUGIN_DIR))


def _load_main():
    spec = importlib.util.spec_from_file_location(
        'document_extraction_plugin_under_test', PLUGIN_DIR / 'main.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plugin = _load_main()


def _request(action, **parameters):
    return {
        'session_id': 'session-1',
        'event': 'request',
        'data': {'action': action, **parameters},
    }


def test_dispatch_extract_returns_descriptor_then_end(tmp_path):
    artifact = tmp_path / 'document.md'
    artifact.write_text('# Extracted')
    result = {
        'schema_version': '1.0',
        'outcome': 'success',
        'result_kind': 'semantic_document',
        'artifact': {
            'path': str(artifact),
            'media_type': 'text/markdown',
            'size_bytes': artifact.stat().st_size,
            'sha256': '0' * 64,
            'role': 'document',
        },
        'provenance': {
            'provider': 'fake',
            'provider_version': '1',
            'canonical_format': 'html',
            'backend': None,
        },
        'metadata': {},
        'warnings': [],
    }
    calls = []

    def extractor(**kwargs):
        calls.append(kwargs)
        return result

    events = plugin.dispatch_request(
        _request(
            'extract',
            path='/input/source.html',
            format='html',
            output_dir=str(tmp_path),
        ),
        extractor=extractor,
    )

    assert [event['data']['type'] for event in events] == ['stream', 'end']
    assert events[0]['data']['data'] == result
    assert '# Extracted' not in json.dumps(events)
    assert calls == [
        {
            'path': '/input/source.html',
            'format_hint': 'html',
            'mime_type': None,
            'intent': 'semantic',
            'output_dir': str(tmp_path),
            'provider': None,
            'fallback': True,
            'provider_options': None,
            'output_format': 'markdown',
        }
    ]


def test_dispatch_capabilities_uses_separate_action():
    snapshot = {
        'schema_version': '1.0',
        'plugin_version': '0.2.0',
        'ready': False,
        'formats': [],
    }
    events = plugin.dispatch_request(
        _request('capabilities'), capabilities=lambda: snapshot
    )
    assert events[0]['data']['data'] == snapshot


def test_dispatch_rejects_unknown_action():
    events = plugin.dispatch_request(_request('delete'))
    error = events[0]['data']['data']
    assert events[0]['data']['type'] == 'error'
    assert error['code'] == 'invalid_request'
    assert error['retryable'] is False


@pytest.mark.parametrize(
    'parameters',
    [
        {'output_dir': '/tmp/output'},
        {'path': '/input/source.html'},
        {'path': '', 'output_dir': '/tmp/output'},
        {'path': '/input/source.html', 'output_dir': ''},
    ],
)
def test_dispatch_requires_path_and_output_dir(parameters):
    events = plugin.dispatch_request(_request('extract', **parameters))

    error = events[0]['data']['data']
    assert events[0]['data']['type'] == 'error'
    assert error['code'] == 'invalid_request'
    assert error['retryable'] is False


def test_dispatch_sanitizes_unexpected_provider_errors(tmp_path):
    def extractor(**kwargs):
        raise RuntimeError('/private/tenant/source.docx failed')

    events = plugin.dispatch_request(
        _request(
            'extract',
            path='/input/source.docx',
            output_dir=str(tmp_path),
        ),
        extractor=extractor,
    )

    error = events[0]['data']['data']
    assert error == {
        'code': 'internal_error',
        'message': 'document extraction provider failed',
        'retryable': False,
    }


def test_manifest_exposes_provider_neutral_actions():
    manifest = yaml.safe_load((PLUGIN_DIR / 'marie-extension.yaml').read_text())
    assert manifest['metadata']['id'] == 'ext.marie.document-extraction'
    assert manifest['metadata']['name'] == 'document-extraction'
    assert {tool['name'] for tool in manifest['tools']} == {
        'capabilities',
        'extract',
    }


def test_wire_schemas_are_valid():
    for path in (PLUGIN_DIR / 'schemas').glob('*.json'):
        Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_runtime_lock_is_exact_and_excludes_docling_ocr_models():
    project = tomllib.loads((PLUGIN_DIR / 'pyproject.toml').read_text())
    lock = tomllib.loads((PLUGIN_DIR / 'uv.lock').read_text())
    dependencies = '\n'.join(project['project']['dependencies']).lower()
    locked_packages = {package['name'].lower() for package in lock['package']}

    assert 'docling-slim[convert-core' in dependencies
    assert '==2.111.0' in dependencies
    assert 'markitdown[docx,pdf,pptx,xlsx]==0.1.6' in dependencies
    assert project['tool']['uv']['package'] is False
    assert {'docling-slim', 'markitdown', 'pydantic'} <= locked_packages
    for forbidden in (
        'rapidocr',
        'easyocr',
        'tesserocr',
        'torch==',
        'torchvision',
        'transformers',
        'docling-ibm-models',
        'docling-parse',
        'playwright',
        'marie-extension',
    ):
        assert forbidden not in locked_packages


def test_plugin_imports_do_not_reference_marie_runtime():
    imports = set()
    paths = [
        PLUGIN_DIR / 'main.py',
        *(PLUGIN_DIR / 'marie_plugins').rglob('*.py'),
    ]
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

    assert not any(name == 'marie' or name.startswith('marie.') for name in imports)
    assert 'marie.extension' not in imports


def test_package_script_includes_plugin_tree_and_schemas(tmp_path):
    result = subprocess.run(
        ['bash', str(PLUGIN_DIR / 'scripts' / 'package.sh'), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    archives = list(tmp_path.glob('marie-plugin-document-extraction_*.zip'))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        names = set(archive.namelist())
    assert 'marie-extension.yaml' in names
    assert 'pyproject.toml' in names
    assert 'uv.lock' in names
    assert 'requirements.in' not in names
    assert 'requirements.txt' not in names
    assert 'marie_plugins/__init__.py' not in names
    assert not any(name.startswith('marie_plugins/runtime/') for name in names)
    assert 'marie_plugins/document_extraction/dispatch.py' in names
    assert 'marie_plugins/document_extraction/providers/docling.py' in names
    assert 'marie_plugins/document_extraction/queries/python-tags.scm' in names
    assert 'marie_plugins/document_extraction/queries/ATTRIBUTION.md' in names
    assert 'schemas/capabilities-v1.json' in names
    assert 'schemas/code-symbols-v1.json' in names
    assert not any(name.startswith('tests/') for name in names)
    assert not any('.venv/' in name for name in names)


def test_extract_input_accepts_tool_parameters_wrapper():
    from marie_plugins.document_extraction.handler import _extract_input

    payload = {'tool_parameters': {'path': '/input/source.html'}, 'user_id': 'u'}
    assert _extract_input(payload) == {'path': '/input/source.html'}


@pytest.mark.parametrize('payload', [None, [], 'value'])
def test_extract_input_rejects_non_mapping(payload):
    from marie_plugins.document_extraction.handler import _extract_input

    assert _extract_input(payload) == {}
