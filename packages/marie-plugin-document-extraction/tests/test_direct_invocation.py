"""In-process checks of the exposed plugin actions — no subprocess, breakpoint-friendly."""

from __future__ import annotations

import hashlib
from pathlib import Path

from marie_plugins.document_extraction.dispatch import extract_document
from marie_plugins.document_extraction.handler import dispatch_request
from marie_plugins.document_extraction.registry import capability_snapshot

FIXTURES = Path(__file__).parent / 'fixtures'


def test_capability_snapshot_direct():
    snapshot = capability_snapshot()
    assert snapshot['ready'] is True
    assert 'pdf' in {item['canonical_format'] for item in snapshot['formats']}


def test_extract_document_direct(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.pdf'),
        output_dir=str(tmp_path),
    )
    assert result['outcome'] == 'success'
    assert result['result_kind'] == 'semantic_document'

    descriptor = result['artifact']
    artifact = Path(descriptor['path'])
    assert artifact.parent == tmp_path
    body = artifact.read_bytes()
    assert len(body) == descriptor['size_bytes']
    assert hashlib.sha256(body).hexdigest() == descriptor['sha256']
    assert body.decode('utf-8').strip()
    assert result['provenance']['canonical_format'] == 'pdf'


def test_dispatch_request_direct(tmp_path):
    frames = dispatch_request(
        {
            'session_id': 's-1',
            'event': 'request',
            'data': {
                'action': 'extract',
                'path': str(FIXTURES / 'sample.pdf'),
                'output_dir': str(tmp_path),
            },
        }
    )
    assert [frame['data']['type'] for frame in frames] == ['stream', 'end']
    result = frames[0]['data']['data']
    assert result['outcome'] == 'success'
    assert Path(result['artifact']['path']).parent == tmp_path


def test_extract_document_honors_provider_selection(tmp_path):
    for provider in ('docling', 'markitdown'):
        result = extract_document(
            path=str(FIXTURES / 'sample.csv'),
            output_dir=str(tmp_path / provider),
            provider=provider,
        )
        assert result['outcome'] == 'success'
        assert result['provenance']['provider'] == provider


def test_extract_document_rejects_unknown_provider(tmp_path):
    frames = dispatch_request(
        {
            'session_id': 's-provider',
            'event': 'request',
            'data': {
                'action': 'extract',
                'path': str(FIXTURES / 'sample.csv'),
                'output_dir': str(tmp_path),
                'provider': 'nope',
            },
        }
    )
    assert len(frames) == 1
    assert frames[0]['data']['type'] == 'error'
    assert frames[0]['data']['data']['code'] == 'invalid_request'


class _FakeProvider:
    output_formats = frozenset({'markdown'})

    def __init__(self, provider_id, *, fail=False):
        self.provider_id = provider_id
        self.fail = fail
        self.calls = 0

    def extract(self, path, canonical_format, options=None, output_format='markdown'):
        from marie_plugins.document_extraction.models import ProviderDocument
        from marie_plugins.document_extraction.providers.base import (
            ProviderUnavailableError,
        )

        self.calls += 1
        if self.fail:
            raise ProviderUnavailableError(f'{self.provider_id} is down')
        return ProviderDocument(
            content=f'# extracted by {self.provider_id}\n',
            provider=self.provider_id,
            provider_version='0',
            metadata={'options': dict(options or {})},
        )


def test_fallback_to_next_provider_by_default(tmp_path, monkeypatch):
    import marie_plugins.document_extraction.dispatch as dispatch_module

    first = _FakeProvider('docling', fail=True)
    second = _FakeProvider('markitdown')
    monkeypatch.setattr(dispatch_module, 'providers_for', lambda fmt: [first, second])

    result = extract_document(
        path=str(FIXTURES / 'sample.csv'),
        output_dir=str(tmp_path),
    )
    assert result['outcome'] == 'success'
    assert result['provenance']['provider'] == 'markitdown'
    assert any('docling' in warning for warning in result['warnings'])
    assert (first.calls, second.calls) == (1, 1)


def test_no_fallback_stops_at_preferred_provider(tmp_path, monkeypatch):
    import marie_plugins.document_extraction.dispatch as dispatch_module

    first = _FakeProvider('docling', fail=True)
    second = _FakeProvider('markitdown')
    monkeypatch.setattr(dispatch_module, 'providers_for', lambda fmt: [first, second])

    result = extract_document(
        path=str(FIXTURES / 'sample.csv'),
        output_dir=str(tmp_path),
        provider='docling',
        fallback=False,
    )
    assert result['outcome'] == 'not_extractable'
    assert result['reason'] == 'providers_exhausted'
    assert result['attempted_providers'] == ['docling']
    assert second.calls == 0


def test_provider_options_reach_the_provider(tmp_path, monkeypatch):
    import marie_plugins.document_extraction.dispatch as dispatch_module

    provider = _FakeProvider('markitdown')
    monkeypatch.setattr(dispatch_module, 'providers_for', lambda fmt: [provider])

    result = extract_document(
        path=str(FIXTURES / 'sample.csv'),
        output_dir=str(tmp_path),
        provider_options={'table_mode': 'accurate', 'max_pages': 5},
    )
    assert result['outcome'] == 'success'
    assert result['metadata']['options'] == {'table_mode': 'accurate', 'max_pages': 5}


def test_provider_options_must_be_an_object(tmp_path):
    frames = dispatch_request(
        {
            'session_id': 's-options',
            'event': 'request',
            'data': {
                'action': 'extract',
                'path': str(FIXTURES / 'sample.csv'),
                'output_dir': str(tmp_path),
                'provider_options': 'not-a-dict',
            },
        }
    )
    assert len(frames) == 1
    assert frames[0]['data']['type'] == 'error'
    assert frames[0]['data']['data']['code'] == 'invalid_request'


def test_output_format_html_and_json_via_docling(tmp_path):
    import json as json_module

    html_result = extract_document(
        path=str(FIXTURES / 'sample.pptx'),
        output_dir=str(tmp_path / 'html'),
        output_format='html',
    )
    assert html_result['outcome'] == 'success'
    assert html_result['artifact']['media_type'] == 'text/html'
    html_path = Path(html_result['artifact']['path'])
    assert html_path.suffix == '.html'
    assert '<' in html_path.read_text()

    json_result = extract_document(
        path=str(FIXTURES / 'sample.pptx'),
        output_dir=str(tmp_path / 'json'),
        output_format='json',
    )
    assert json_result['outcome'] == 'success'
    assert json_result['result_kind'] == 'structured_document'
    assert json_result['artifact']['media_type'] == 'application/json'
    body = json_module.loads(Path(json_result['artifact']['path']).read_text())
    assert isinstance(body, dict)


def test_output_format_unsupported_by_all_providers(tmp_path):
    result = extract_document(
        path=str(FIXTURES / 'sample.pdf'),
        output_dir=str(tmp_path),
        output_format='html',
    )
    assert result['outcome'] == 'not_extractable'
    assert result['reason'] == 'no_ready_provider'
    assert any('html' in warning for warning in result['warnings'])


def test_unknown_output_format_is_invalid_request(tmp_path):
    frames = dispatch_request(
        {
            'session_id': 's-fmt',
            'event': 'request',
            'data': {
                'action': 'extract',
                'path': str(FIXTURES / 'sample.csv'),
                'output_dir': str(tmp_path),
                'output_format': 'docx',
            },
        }
    )
    assert len(frames) == 1
    assert frames[0]['data']['type'] == 'error'
    assert frames[0]['data']['data']['code'] == 'invalid_request'


def test_dispatch_request_missing_input_direct(tmp_path):
    frames = dispatch_request(
        {
            'session_id': 's-err',
            'event': 'request',
            'data': {
                'action': 'extract',
                'path': str(FIXTURES / 'does-not-exist.pdf'),
                'output_dir': str(tmp_path),
            },
        }
    )
    assert len(frames) == 1
    assert frames[0]['data']['type'] == 'error'
    error = frames[0]['data']['data']
    assert error['code'] == 'invalid_request'
    assert error['retryable'] is False
