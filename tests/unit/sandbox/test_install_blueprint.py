"""
Unit tests for marie.sandbox.install_blueprint.

Uses unittest.mock to stub out all HTTP calls and tarfile I/O — no network
or real gateway needed.
"""

from __future__ import annotations

import io
import json
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from marie.excepts import BadConfigSource
from marie.sandbox.install_blueprint import (
    BlueprintInstallResult,
    _download_blueprint,
    _extract_manifest,
    install_blueprint,
)

# ----------------------------------------------------------------- helpers ---

_GATEWAY = 'http://sbx-test-server:51000'
_API_KEY = 'mas_' + 'x' * 54
_BP_ID = 'ner-vlm-ocr-entity-extraction'
_REGISTRY = 'https://blueprints.example.com'


def _make_archive(manifest: dict, extra_files: dict[str, str] | None = None) -> bytes:
    """Build a minimal .blueprint tar.gz in memory."""
    import yaml  # type: ignore[import]

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        manifest_bytes = yaml.dump(manifest).encode()
        info = tarfile.TarInfo('blueprint.yaml')
        info.size = len(manifest_bytes)
        tf.addfile(info, io.BytesIO(manifest_bytes))

        for path, content in (extra_files or {}).items():
            content_bytes = content.encode()
            finfo = tarfile.TarInfo(path)
            finfo.size = len(content_bytes)
            tf.addfile(finfo, io.BytesIO(content_bytes))

    return buf.getvalue()


def _empty_manifest(bp_id: str = _BP_ID) -> dict:
    return {
        'manifestVersion': 1,
        'id': bp_id,
        'name': 'Test Blueprint',
        'version': '1.0.0',
        'artifacts': [],
    }


# --------------------------------------------------------- unit: helpers ---


class TestExtractManifest:
    def test_extracts_blueprint_yaml(self):
        archive = _make_archive(_empty_manifest())
        manifest = _extract_manifest(archive)
        assert manifest['id'] == _BP_ID

    def test_raises_when_no_blueprint_yaml(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz'):
            pass
        with pytest.raises(ValueError, match='blueprint.yaml not found'):
            _extract_manifest(buf.getvalue())


class TestDownloadBlueprint:
    def test_constructs_correct_url(self):
        with patch('marie.sandbox.install_blueprint._http_get') as mock_get:
            mock_get.return_value = b'data'
            _download_blueprint('https://reg.example.com', 'my-bp')
            mock_get.assert_called_once_with('https://reg.example.com/my-bp.blueprint')

    def test_strips_trailing_slash_from_registry(self):
        with patch('marie.sandbox.install_blueprint._http_get') as mock_get:
            mock_get.return_value = b'data'
            _download_blueprint('https://reg.example.com/', 'my-bp')
            mock_get.assert_called_once_with('https://reg.example.com/my-bp.blueprint')

    def test_propagates_download_error(self):
        with patch('marie.sandbox.install_blueprint._http_get') as mock_get:
            mock_get.side_effect = RuntimeError('connection refused')
            with pytest.raises(RuntimeError, match='connection refused'):
                _download_blueprint(_REGISTRY, _BP_ID)


# -------------------------------------------------------- unit: install ---


class TestInstallBlueprint:
    def _run(
        self,
        archive_bytes: bytes | None = None,
        gateway: str = _GATEWAY,
        api_key: str = _API_KEY,
        bp_id: str = _BP_ID,
        registry: str = _REGISTRY,
    ) -> BlueprintInstallResult:
        if archive_bytes is None:
            archive_bytes = _make_archive(_empty_manifest())
        with (
            patch('marie.sandbox.install_blueprint._download_blueprint', return_value=archive_bytes),
            patch('marie.sandbox.install_blueprint._http_post_json', return_value={'success': True}),
        ):
            return install_blueprint(gateway, api_key, bp_id, registry)

    # -- validation --

    def test_raises_on_empty_gateway_url(self):
        with pytest.raises(BadConfigSource, match='gateway_url'):
            install_blueprint('', _API_KEY, _BP_ID, _REGISTRY)

    def test_raises_on_empty_api_key(self):
        with pytest.raises(BadConfigSource, match='api_key'):
            install_blueprint(_GATEWAY, '', _BP_ID, _REGISTRY)

    def test_raises_on_empty_blueprint_id(self):
        with pytest.raises(BadConfigSource, match='blueprint_id'):
            install_blueprint(_GATEWAY, _API_KEY, '', _REGISTRY)

    def test_raises_on_empty_registry_url(self):
        with pytest.raises(BadConfigSource, match='registry_url'):
            install_blueprint(_GATEWAY, _API_KEY, _BP_ID, '')

    # -- happy path --

    def test_returns_result_with_blueprint_id(self):
        result = self._run()
        assert result.blueprint_id == _BP_ID

    def test_empty_blueprint_succeeds_with_no_installs(self):
        result = self._run()
        assert result.success
        assert result.connectors_installed == []
        assert result.planners_registered == []
        assert result.errors == []

    def test_installs_connector_artifact(self):
        manifest = {
            **_empty_manifest(),
            'artifacts': [
                {
                    'kind': 'connector',
                    'ref': 'connector.my-ocr',
                    'path': 'connectors/my-ocr/',
                }
            ],
        }
        archive = _make_archive(
            manifest,
            extra_files={'connectors/my-ocr/connector.yaml': 'id: connector.my-ocr\n'},
        )
        result = self._run(archive_bytes=archive)
        assert result.success
        assert 'connector.my-ocr' in result.connectors_installed

    def test_skips_unsupported_artifact_kinds(self):
        manifest = {
            **_empty_manifest(),
            'artifacts': [
                {'kind': 'prompt_package', 'ref': 'p1', 'path': 'prompts/p1/'},
                {'kind': 'rag_index', 'ref': 'r1', 'path': 'rag/r1/'},
                {'kind': 'sample_data', 'ref': 's1'},
            ],
        }
        archive = _make_archive(manifest)
        result = self._run(archive_bytes=archive)
        assert result.success
        assert sorted(result.skipped_kinds) == ['prompt_package', 'rag_index', 'sample_data']
        assert result.connectors_installed == []

    def test_download_failure_captured_in_errors(self):
        with patch(
            'marie.sandbox.install_blueprint._download_blueprint',
            side_effect=RuntimeError('registry unreachable'),
        ):
            result = install_blueprint(_GATEWAY, _API_KEY, _BP_ID, _REGISTRY)
        assert not result.success
        assert any('registry unreachable' in e for e in result.errors)

    def test_gateway_failure_captured_in_errors(self):
        manifest = {
            **_empty_manifest(),
            'artifacts': [{'kind': 'connector', 'ref': 'c1', 'path': 'connectors/c1/'}],
        }
        archive = _make_archive(
            manifest, extra_files={'connectors/c1/connector.yaml': 'id: c1\n'}
        )
        with (
            patch('marie.sandbox.install_blueprint._download_blueprint', return_value=archive),
            patch(
                'marie.sandbox.install_blueprint._http_post_json',
                side_effect=RuntimeError('gateway 500'),
            ),
        ):
            result = install_blueprint(_GATEWAY, _API_KEY, _BP_ID, _REGISTRY)
        assert not result.success
        assert any('gateway 500' in e for e in result.errors)

    def test_success_attribute_false_when_errors(self):
        with patch(
            'marie.sandbox.install_blueprint._download_blueprint',
            side_effect=RuntimeError('boom'),
        ):
            result = install_blueprint(_GATEWAY, _API_KEY, _BP_ID, _REGISTRY)
        assert result.success is False

    def test_idempotent_conflict_is_not_an_error(self):
        """A 409 from the gateway (already installed) must not count as an error."""
        manifest = {
            **_empty_manifest(),
            'artifacts': [{'kind': 'connector', 'ref': 'c1', 'path': 'connectors/c1/'}],
        }
        archive = _make_archive(
            manifest, extra_files={'connectors/c1/connector.yaml': 'id: c1\n'}
        )
        with (
            patch('marie.sandbox.install_blueprint._download_blueprint', return_value=archive),
            patch(
                'marie.sandbox.install_blueprint._http_post_json',
                return_value={'success': True, 'conflict': True},
            ),
        ):
            result = install_blueprint(_GATEWAY, _API_KEY, _BP_ID, _REGISTRY)
        assert result.success
        assert 'c1' in result.connectors_installed
