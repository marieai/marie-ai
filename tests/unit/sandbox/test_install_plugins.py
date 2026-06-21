"""
Unit tests for marie.sandbox.install_plugins.

Uses unittest.mock to stub out HTTP calls — no network or real gateway needed.
"""

from __future__ import annotations

import io
import tarfile
from unittest.mock import patch

import pytest

from marie.excepts import BadConfigSource
from marie.sandbox.install_plugins import (
    PluginInstallResult,
    PluginRef,
    _has_deferred_credentials,
    install_plugins,
)

# ----------------------------------------------------------------- helpers ---

_GATEWAY = 'http://sbx-test-server:51000'
_API_KEY = 'mas_' + 'x' * 54
_REGISTRY = 'https://plugins.example.com'

_REF1 = {'packageId': 'connector.ocr-engine', 'version': '2.1.0'}
_REF2 = {'packageId': 'connector.ner-model', 'version': '1.4.2'}


def _make_plugin_archive(files: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for path, content in (files or {'connector.yaml': 'id: test\n'}).items():
            b = content.encode()
            info = tarfile.TarInfo(path)
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    return buf.getvalue()


# --------------------------------------------------------- unit: PluginRef ---


class TestPluginRef:
    def test_from_dict_minimal(self):
        ref = PluginRef.from_dict({'packageId': 'p1', 'version': '1.0.0'})
        assert ref.package_id == 'p1'
        assert ref.version == '1.0.0'
        assert ref.install_mode == 'install-from-registry'

    def test_from_dict_with_install_mode(self):
        ref = PluginRef.from_dict(
            {'packageId': 'p1', 'version': '1.0.0', 'installMode': 'require-installed'}
        )
        assert ref.install_mode == 'require-installed'

    def test_from_dict_missing_package_id(self):
        with pytest.raises(KeyError):
            PluginRef.from_dict({'version': '1.0.0'})

    def test_has_deferred_credentials_false_when_no_bindings(self):
        ref = PluginRef('p1', '1.0.0')
        assert not _has_deferred_credentials(ref)

    def test_has_deferred_credentials_true_when_deferred(self):
        ref = PluginRef(
            'p1',
            '1.0.0',
            credential_bindings=[{'key': 'API_KEY', 'bindingMode': 'deferred'}],
        )
        assert _has_deferred_credentials(ref)

    def test_has_deferred_credentials_false_when_install_time(self):
        ref = PluginRef(
            'p1',
            '1.0.0',
            credential_bindings=[{'key': 'API_KEY', 'bindingMode': 'install-time'}],
        )
        assert not _has_deferred_credentials(ref)


# -------------------------------------------------- unit: install_plugins ---


class TestInstallPlugins:
    def _run(
        self,
        refs: list[dict] | None = None,
        gateway: str = _GATEWAY,
        api_key: str = _API_KEY,
        registry: str = _REGISTRY,
    ) -> PluginInstallResult:
        archive = _make_plugin_archive()
        with (
            patch('marie.sandbox.install_plugins._http_get', return_value=archive),
            patch('marie.sandbox.install_plugins._http_post_json', return_value={'success': True}),
        ):
            return install_plugins(gateway, api_key, refs or [_REF1], registry)

    # -- validation --

    def test_raises_on_empty_gateway_url(self):
        with pytest.raises(BadConfigSource, match='gateway_url'):
            install_plugins('', _API_KEY, [_REF1], _REGISTRY)

    def test_raises_on_empty_api_key(self):
        with pytest.raises(BadConfigSource, match='api_key'):
            install_plugins(_GATEWAY, '', [_REF1], _REGISTRY)

    def test_raises_on_empty_registry_url(self):
        with pytest.raises(BadConfigSource, match='registry_url'):
            install_plugins(_GATEWAY, _API_KEY, [_REF1], '')

    # -- no-ops --

    def test_empty_refs_returns_empty_success(self):
        result = install_plugins(_GATEWAY, _API_KEY, [], _REGISTRY)
        assert result.success
        assert result.installed == []
        assert result.errors == []

    # -- happy path --

    def test_installs_single_ref(self):
        result = self._run(refs=[_REF1])
        assert result.success
        assert 'connector.ocr-engine@2.1.0' in result.installed

    def test_installs_multiple_refs(self):
        result = self._run(refs=[_REF1, _REF2])
        assert result.success
        assert len(result.installed) == 2

    def test_requests_correct_download_url(self):
        with (
            patch('marie.sandbox.install_plugins._http_get') as mock_get,
            patch('marie.sandbox.install_plugins._http_post_json', return_value={'success': True}),
        ):
            mock_get.return_value = _make_plugin_archive()
            install_plugins(_GATEWAY, _API_KEY, [_REF1], _REGISTRY)
            expected_url = f'{_REGISTRY}/connector.ocr-engine/2.1.0.plugin'
            mock_get.assert_called_once_with(expected_url)

    def test_deferred_cred_ref_in_deferred_list(self):
        ref_with_deferred = {
            'packageId': 'connector.llm',
            'version': '1.0.0',
            'credentialBindings': [{'key': 'API_KEY', 'bindingMode': 'deferred'}],
        }
        result = self._run(refs=[ref_with_deferred])
        assert result.success
        assert 'connector.llm@1.0.0' in result.deferred
        assert 'connector.llm@1.0.0' in result.installed

    # -- error handling --

    def test_download_error_captured_per_ref(self):
        with (
            patch(
                'marie.sandbox.install_plugins._http_get',
                side_effect=RuntimeError('registry 503'),
            ),
        ):
            result = install_plugins(_GATEWAY, _API_KEY, [_REF1], _REGISTRY)
        assert not result.success
        assert any('registry 503' in e for e in result.errors)
        assert result.installed == []

    def test_gateway_error_captured_per_ref(self):
        archive = _make_plugin_archive()
        with (
            patch('marie.sandbox.install_plugins._http_get', return_value=archive),
            patch(
                'marie.sandbox.install_plugins._http_post_json',
                side_effect=RuntimeError('gateway 500'),
            ),
        ):
            result = install_plugins(_GATEWAY, _API_KEY, [_REF1], _REGISTRY)
        assert not result.success
        assert any('gateway 500' in e for e in result.errors)

    def test_invalid_ref_dict_captured_in_errors(self):
        bad_refs = [{'version': '1.0.0'}]  # missing packageId
        result = install_plugins(_GATEWAY, _API_KEY, bad_refs, _REGISTRY)
        assert not result.success
        assert result.errors

    def test_one_failure_does_not_stop_other_refs(self):
        """A failure on ref-1 must not skip ref-2."""
        archive = _make_plugin_archive()
        call_count = [0]

        def _get_side_effect(url: str) -> bytes:
            call_count[0] += 1
            if 'ocr-engine' in url:
                raise RuntimeError('ocr-engine registry error')
            return archive

        with (
            patch('marie.sandbox.install_plugins._http_get', side_effect=_get_side_effect),
            patch('marie.sandbox.install_plugins._http_post_json', return_value={'success': True}),
        ):
            result = install_plugins(_GATEWAY, _API_KEY, [_REF1, _REF2], _REGISTRY)

        assert 'connector.ner-model@1.4.2' in result.installed
        assert len(result.errors) == 1

    def test_conflict_409_treated_as_success(self):
        archive = _make_plugin_archive()
        with (
            patch('marie.sandbox.install_plugins._http_get', return_value=archive),
            patch(
                'marie.sandbox.install_plugins._http_post_json',
                return_value={'success': True, 'conflict': True},
            ),
        ):
            result = install_plugins(_GATEWAY, _API_KEY, [_REF1], _REGISTRY)
        assert result.success
        assert 'connector.ocr-engine@2.1.0' in result.installed
