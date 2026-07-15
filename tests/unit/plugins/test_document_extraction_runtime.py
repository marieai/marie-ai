import hashlib
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = REPO_ROOT / 'packages' / 'marie-plugin-document-extraction'
RUNTIME_DIR = REPO_ROOT / 'packages' / 'marie-plugin-daemon' / 'python_runtime'
sys.path.insert(0, str(RUNTIME_DIR))
sys.path.insert(0, str(PLUGIN_DIR))

from marie_plugins.document_extraction import dispatch  # noqa: E402
from marie_plugins.document_extraction.artifacts import (  # noqa: E402
    write_document_artifact,
)
from marie_plugins.document_extraction.detection import detect_format  # noqa: E402
from marie_plugins.document_extraction.models import ProviderDocument  # noqa: E402
from marie_plugins.document_extraction.providers.base import (  # noqa: E402
    ProviderNotExtractableError,
)
from marie_plugins.document_extraction.registry import (  # noqa: E402
    capability_snapshot_model,
)


class _FakeProvider:
    provider_id = 'fake'
    formats = frozenset({'html'})
    output_formats = frozenset({'markdown'})

    def is_ready(self, canonical_format):
        return canonical_format == 'html'

    def extract(self, path, canonical_format, options=None, output_format='markdown'):
        return ProviderDocument(
            content='# Heading\n\nBody',
            provider=self.provider_id,
            provider_version='1.2.3',
            backend='FakeBackend',
        )


def test_detection_uses_content_and_extension(tmp_path):
    path = tmp_path / 'source.html'
    path.write_text('<!doctype html><html><body>text</body></html>')
    result = detect_format(str(path), mime_type='text/html')
    assert result.canonical_format == 'html'
    assert set(result.evidence) == {'content', 'extension', 'mime'}


def test_detection_rejects_conflicting_evidence(tmp_path):
    path = tmp_path / 'source.pdf'
    path.write_text('<html><body>text</body></html>')
    with pytest.raises(ValueError, match='Conflicting format evidence'):
        detect_format(str(path), format_hint='pdf')


def test_capability_snapshot_contains_only_ready_edges():
    snapshot = capability_snapshot_model([_FakeProvider()])
    assert snapshot.ready is True
    assert [item.canonical_format for item in snapshot.formats] == ['html']
    assert snapshot.formats[0].providers == ['fake']


def test_dispatch_writes_body_to_artifact(monkeypatch, tmp_path):
    path = tmp_path / 'source.html'
    path.write_text('<html><body>text</body></html>')
    output_dir = tmp_path / 'output'
    monkeypatch.setattr(dispatch, 'providers_for', lambda _: [_FakeProvider()])

    result = dispatch.extract_document(path=str(path), output_dir=str(output_dir))

    assert result['outcome'] == 'success'
    artifact = Path(result['artifact']['path'])
    assert artifact.parent == output_dir.resolve()
    assert artifact.read_text() == '# Heading\n\nBody'
    data = artifact.read_bytes()
    assert result['artifact']['sha256'] == hashlib.sha256(data).hexdigest()
    assert result['artifact']['size_bytes'] == len(data)
    assert '# Heading' not in json.dumps(result)

    schema = json.loads(
        (PLUGIN_DIR / 'schemas' / 'extraction-response-v1.json').read_text()
    )
    Draft202012Validator(schema).validate(result)


def test_large_document_result_stays_out_of_protocol_json(monkeypatch, tmp_path):
    path = tmp_path / 'source.html'
    path.write_text('<html><body>text</body></html>')
    content = '\n'.join(f'# Page {page}\n' + ('x' * 10_000) for page in range(101))
    provider = _FakeProvider()
    provider.extract = lambda *_, **__: ProviderDocument(
        content=content,
        provider='fake',
        provider_version='1.2.3',
        backend='FakeBackend',
    )
    monkeypatch.setattr(dispatch, 'providers_for', lambda _: [provider])

    result = dispatch.extract_document(
        path=str(path), output_dir=str(tmp_path / 'output')
    )

    encoded = json.dumps(result)
    assert len(encoded) < 2_000
    assert 'x' * 1_000 not in encoded
    assert Path(result['artifact']['path']).read_text() == content


def test_artifact_is_read_only(tmp_path):
    artifact = write_document_artifact(
        'body', output_dir=str(tmp_path), media_type='text/markdown'
    )
    assert Path(artifact.path).stat().st_mode & 0o222 == 0


def test_dispatch_returns_not_extractable_without_ready_provider(tmp_path):
    path = tmp_path / 'source.md'
    path.write_text('# Heading')

    result = dispatch.extract_document(path=str(path), format_hint='markdown')

    assert result['outcome'] == 'not_extractable'
    assert result['reason'] == 'no_ready_provider'


def test_dispatch_falls_back_only_after_classified_provider_result(
    monkeypatch, tmp_path
):
    path = tmp_path / 'source.html'
    path.write_text('<html><body>text</body></html>')

    class NotExtractableProvider(_FakeProvider):
        provider_id = 'first'

        def extract(self, path, canonical_format, options=None, output_format='markdown'):
            raise ProviderNotExtractableError('no useful content')

    monkeypatch.setattr(
        dispatch,
        'providers_for',
        lambda _: [NotExtractableProvider(), _FakeProvider()],
    )

    result = dispatch.extract_document(
        path=str(path), output_dir=str(tmp_path / 'output')
    )

    assert result['outcome'] == 'success'
    assert result['provenance']['provider'] == 'fake'
    assert result['warnings'] == ['first: no useful content']


def test_dispatch_does_not_mask_unexpected_provider_failure(monkeypatch, tmp_path):
    path = tmp_path / 'source.html'
    path.write_text('<html><body>text</body></html>')

    class BrokenProvider(_FakeProvider):
        provider_id = 'broken'

        def extract(self, path, canonical_format, options=None, output_format='markdown'):
            raise RuntimeError('provider bug')

    monkeypatch.setattr(
        dispatch, 'providers_for', lambda _: [BrokenProvider(), _FakeProvider()]
    )

    with pytest.raises(RuntimeError, match='provider bug'):
        dispatch.extract_document(
            path=str(path), output_dir=str(tmp_path / 'output')
        )
