"""Unit tests for the sandbox blueprint-import and plugin-install service layer.

Coverage:
  - Payload parsing / validation (request body shapes).
  - Per-artifact dispatch: query_plan applied; all deferred kinds correctly deferred.
  - Partial-result contract: status field reflects applied/deferred mix.
  - Idempotency: re-importing the same query_plan does not raise.
  - Auth rejection is exercised at the TokenBearer level (tested separately in
    tests/auth/); here we only test the service and registry in isolation.
  - Plugin install: already-registered connector → 'installed';
    unknown plugin → 'deferred' (honest report of dify-parity gap).

All tests run without a live Postgres, MinIO, or ETCD.  External I/O is
mocked only where unavoidable (ConnectorRegistry, QueryPlanRegistry state).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from marie.sandbox.blueprints.models import (
    ArtifactResult,
    BlueprintImportResponse,
    PluginInstallResponse,
)
from marie.sandbox.blueprints.registry import BlueprintRegistry
from marie.sandbox.blueprints.service import BlueprintImportService, install_plugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_query_plan_artifact(
    ref: str = 'workflow/test-plan',
    name: str = 'Test Plan',
    plan_definition: dict | None = None,
) -> dict[str, Any]:
    return {
        'kind': 'query_plan',
        'ref': ref,
        'create': {
            'name': name,
            'description': 'A test query plan',
            'version': '1.0.0',
            'tags': ['test'],
            'category': 'testing',
            'planDefinition': plan_definition or {'nodes': []},
        },
    }


def _make_manifest(artifacts: list[dict]) -> dict[str, Any]:
    return {
        'manifestVersion': 1,
        'id': 'bp.test',
        'name': 'Test Blueprint',
        'version': '0.1.0',
        'parameters': [],
        'artifacts': artifacts,
        'install': {'conflictPolicy': 'fail'},
    }


# ---------------------------------------------------------------------------
# BlueprintRegistry
# ---------------------------------------------------------------------------


class TestBlueprintRegistry:
    def test_lookup_missing_returns_none(self, tmp_path: Path) -> None:
        registry = BlueprintRegistry(blueprints_dir=str(tmp_path))
        assert registry.lookup('nonexistent-blueprint') is None

    def test_lookup_yaml_file(self, tmp_path: Path) -> None:
        bp_file = tmp_path / 'my-blueprint.yaml'
        bp_file.write_text('id: my-blueprint\nname: My Blueprint\nartifacts: []\n')
        registry = BlueprintRegistry(blueprints_dir=str(tmp_path))
        result = registry.lookup('my-blueprint')
        assert result is not None
        assert result['id'] == 'my-blueprint'

    def test_lookup_yml_extension(self, tmp_path: Path) -> None:
        bp_file = tmp_path / 'my-blueprint.yml'
        bp_file.write_text('id: my-blueprint\nname: My Blueprint\nartifacts: []\n')
        registry = BlueprintRegistry(blueprints_dir=str(tmp_path))
        result = registry.lookup('my-blueprint')
        assert result is not None

    def test_lookup_invalid_yaml_returns_none(self, tmp_path: Path) -> None:
        bp_file = tmp_path / 'bad.yaml'
        bp_file.write_text('- not: a\n  mapping: at root\n  but: actually valid\n')
        # A list at YAML root is not a dict — registry should reject it.
        registry = BlueprintRegistry(blueprints_dir=str(tmp_path))
        result = registry.lookup('bad')
        assert result is None

    def test_env_var_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bp_file = tmp_path / 'env-bp.yaml'
        bp_file.write_text('id: env-bp\nartifacts: []\n')
        monkeypatch.setenv('MARIE_BLUEPRINTS_DIR', str(tmp_path))
        # Pass no explicit dir — should read from env var.
        registry = BlueprintRegistry()
        result = registry.lookup('env-bp')
        assert result is not None
        assert result['id'] == 'env-bp'


# ---------------------------------------------------------------------------
# BlueprintImportService — payload parsing
# ---------------------------------------------------------------------------


class TestBlueprintImportServicePayloadParsing:
    def test_non_list_artifacts_returns_failed(self) -> None:
        svc = BlueprintImportService()
        resp = svc.import_blueprint('bp.test', {'artifacts': 'not-a-list'})
        assert resp.status == 'failed'
        assert 'not a list' in (resp.message or '')

    def test_empty_artifacts_returns_completed(self) -> None:
        svc = BlueprintImportService()
        resp = svc.import_blueprint('bp.empty', _make_manifest([]))
        assert resp.status == 'completed'
        assert resp.applied == []
        assert resp.deferred == []
        assert resp.failed == []

    def test_non_dict_artifact_entries_are_skipped(self) -> None:
        svc = BlueprintImportService()
        resp = svc.import_blueprint('bp.test', {'artifacts': ['not-a-dict', 42, None]})
        # No applied, no deferred, no failed → completed (nothing to do).
        assert resp.status == 'completed'


# ---------------------------------------------------------------------------
# BlueprintImportService — query_plan (APPLIED path)
# ---------------------------------------------------------------------------


class TestQueryPlanArtifact:
    def _fresh_service(self) -> BlueprintImportService:
        return BlueprintImportService()

    @patch('marie.sandbox.blueprints.service.QueryPlanRegistry')
    def test_query_plan_applied_when_registration_succeeds(self, mock_registry: MagicMock) -> None:
        mock_registry.get_metadata.return_value = None  # not registered yet
        mock_registry.register_from_json.return_value = True

        svc = self._fresh_service()
        artifact = _make_query_plan_artifact(plan_definition={'nodes': [{'id': 'n1'}]})
        resp = svc.import_blueprint('bp.test', _make_manifest([artifact]))

        assert resp.status == 'completed'
        assert len(resp.applied) == 1
        assert resp.applied[0].kind == 'query_plan'
        assert resp.applied[0].status == 'applied'
        mock_registry.register_from_json.assert_called_once()

    @patch('marie.sandbox.blueprints.service.QueryPlanRegistry')
    def test_query_plan_idempotent_already_registered(self, mock_registry: MagicMock) -> None:
        existing_meta = MagicMock()
        mock_registry.get_metadata.return_value = existing_meta  # already present

        svc = self._fresh_service()
        artifact = _make_query_plan_artifact()
        resp = svc.import_blueprint('bp.test', _make_manifest([artifact]))

        assert resp.status == 'completed'
        assert resp.applied[0].status == 'applied'
        assert 'idempotent' in (resp.applied[0].reason or '')
        # Should NOT call register again.
        mock_registry.register_from_json.assert_not_called()

    @patch('marie.sandbox.blueprints.service.QueryPlanRegistry')
    def test_query_plan_failed_when_registration_returns_false(self, mock_registry: MagicMock) -> None:
        mock_registry.get_metadata.return_value = None
        mock_registry.register_from_json.return_value = False

        svc = self._fresh_service()
        artifact = _make_query_plan_artifact()
        resp = svc.import_blueprint('bp.test', _make_manifest([artifact]))

        assert resp.status == 'failed'
        assert len(resp.failed) == 1
        assert resp.failed[0].status == 'failed'

    def test_query_plan_deferred_when_no_inline_plan_definition(self) -> None:
        svc = self._fresh_service()
        artifact = {
            'kind': 'query_plan',
            'ref': 'workflow/archive-only',
            'create': {
                'name': 'Archive Plan',
                'planDefinitionPath': 'workflows/archive-plan.json',
            },
        }
        resp = svc.import_blueprint('bp.test', _make_manifest([artifact]))

        assert resp.status == 'partial'
        assert len(resp.deferred) == 1
        assert 'planDefinitionPath' in (resp.deferred[0].reason or '')


# ---------------------------------------------------------------------------
# BlueprintImportService — deferred kinds
# ---------------------------------------------------------------------------


class TestDeferredArtifactKinds:
    _DEFERRED_KINDS = [
        'tenant_registry',
        'rag_index',
        'rag_source',
        'prompt_package',
        'script_package',
        'webapp_package',
        'prefab',
        'skill',
        'agent',
    ]

    @pytest.mark.parametrize('kind', _DEFERRED_KINDS)
    def test_deferred_kind_returns_deferred_result(self, kind: str) -> None:
        svc = BlueprintImportService()
        artifact = {'kind': kind, 'ref': f'{kind}/test-artifact'}
        resp = svc.import_blueprint('bp.test', _make_manifest([artifact]))

        assert len(resp.deferred) == 1
        result = resp.deferred[0]
        assert result.status == 'deferred'
        assert result.kind == kind
        assert result.reason is not None and len(result.reason) > 0

    def test_unknown_kind_is_also_deferred(self) -> None:
        svc = BlueprintImportService()
        artifact = {'kind': 'future_kind', 'ref': 'future/artifact'}
        resp = svc.import_blueprint('bp.test', _make_manifest([artifact]))

        assert len(resp.deferred) == 1
        assert resp.deferred[0].kind == 'future_kind'

    def test_sample_data_is_applied(self) -> None:
        svc = BlueprintImportService()
        artifact = {'kind': 'sample_data', 'ref': 'data/sample', 'path': 'sample-data/sample.json'}
        resp = svc.import_blueprint('bp.test', _make_manifest([artifact]))

        assert len(resp.applied) == 1
        assert resp.applied[0].status == 'applied'


# ---------------------------------------------------------------------------
# BlueprintImportService — partial result contract
# ---------------------------------------------------------------------------


class TestPartialResultContract:
    @patch('marie.sandbox.blueprints.service.QueryPlanRegistry')
    def test_partial_status_when_some_applied_some_deferred(self, mock_registry: MagicMock) -> None:
        mock_registry.get_metadata.return_value = None
        mock_registry.register_from_json.return_value = True

        svc = BlueprintImportService()
        artifacts = [
            _make_query_plan_artifact(ref='workflow/plan-a', name='Plan A'),
            {'kind': 'rag_index', 'ref': 'rag-index/docs'},
            {'kind': 'tenant_registry', 'ref': 'tenant/demo'},
        ]
        resp = svc.import_blueprint('bp.test', _make_manifest(artifacts))

        assert resp.status == 'partial'
        assert len(resp.applied) == 1
        assert len(resp.deferred) == 2
        assert len(resp.failed) == 0
        # blueprint_id is preserved.
        assert resp.blueprint_id == 'bp.test'

    def test_response_model_is_serialisable(self) -> None:
        resp = BlueprintImportResponse(
            blueprint_id='bp.test',
            status='partial',
            applied=[ArtifactResult(ref='r1', kind='query_plan', status='applied')],
            deferred=[ArtifactResult(ref='r2', kind='rag_index', status='deferred', reason='no backend')],
            failed=[],
        )
        d = resp.model_dump()
        assert d['blueprint_id'] == 'bp.test'
        assert d['status'] == 'partial'
        assert d['applied'][0]['status'] == 'applied'
        assert d['deferred'][0]['reason'] == 'no backend'


# ---------------------------------------------------------------------------
# Plugin install
# ---------------------------------------------------------------------------


class TestPluginInstall:
    @patch('marie.sandbox.blueprints.service.ConnectorRegistry')
    def test_already_registered_connector_returns_installed(self, mock_cr: MagicMock) -> None:
        mock_cr.get.return_value = MagicMock()  # connector found

        result = install_plugin('connector.ocr-engine', '2.1.0')

        assert result['status'] == 'installed'
        assert result['package_id'] == 'connector.ocr-engine'
        assert result['version'] == '2.1.0'

    @patch('marie.sandbox.blueprints.service.ConnectorRegistry')
    def test_unknown_plugin_returns_deferred(self, mock_cr: MagicMock) -> None:
        mock_cr.get.return_value = None  # not found

        result = install_plugin('connector.unknown-plugin', '1.0.0')

        assert result['status'] == 'deferred'
        assert 'dify-parity' in result.get('message', '')

    @patch('marie.sandbox.blueprints.service.ConnectorRegistry')
    def test_plugin_install_response_shape(self, mock_cr: MagicMock) -> None:
        mock_cr.get.return_value = None

        result = install_plugin('connector.ner-model', '1.4.2')

        assert 'package_id' in result
        assert 'version' in result
        assert 'status' in result
        assert result['status'] in ('installed', 'deferred', 'failed')


# ---------------------------------------------------------------------------
# Multi-artifact import — full NER VLM blueprint fixture
# ---------------------------------------------------------------------------


class TestNerVlmBlueprintFixture:
    """Smoke-test the NER VLM OCR blueprint sample from analysis/blueprint/samples/."""

    _FIXTURE: dict[str, Any] = {
        'manifestVersion': 1,
        'id': 'bp.ner-vlm-ocr-entity-extraction',
        'name': 'VLM OCR Entity Extraction',
        'version': '0.1.0',
        'parameters': [],
        'artifacts': [
            {'kind': 'tenant_registry', 'ref': 'tenant/demo', 'path': 'tenants/tenants.yaml'},
            {'kind': 'prompt_package', 'ref': 'prompt/vlm-ocr-entity', 'path': 'prompts/'},
            {'kind': 'rag_index', 'ref': 'rag-index/entity-examples'},
            {'kind': 'rag_source', 'ref': 'rag-source/retrieved-examples'},
            {
                'kind': 'webapp_package',
                'ref': 'webapp/entity-review-plugin',
                'path': 'webapps/entity-review-plugin/',
                'create': {'slug': 'ner-vlm-entity-review', 'name': 'NER Entity Review'},
            },
            {
                'kind': 'query_plan',
                'ref': 'workflow/vlm-ocr-entity-extraction',
                'path': 'workflows/vlm-ocr-entity-extraction.json',
                'create': {
                    'name': 'VLM OCR Entity Extraction',
                    'version': '0.1.0',
                    'planDefinitionPath': 'workflows/vlm-ocr-entity-extraction.json',
                },
            },
            {'kind': 'sample_data', 'ref': 'data/hello-world-input', 'path': 'sample-data/sample-input.json'},
            {'kind': 'sample_data', 'ref': 'data/expected-entities', 'path': 'sample-data/expected-entities.json'},
        ],
        'install': {'conflictPolicy': 'prompt'},
    }

    def test_fixture_import_produces_expected_deferred_kinds(self) -> None:
        svc = BlueprintImportService()
        resp = svc.import_blueprint('bp.ner-vlm-ocr-entity-extraction', self._FIXTURE)

        deferred_kinds = {r.kind for r in resp.deferred}
        applied_kinds = {r.kind for r in resp.applied}

        # query_plan is deferred because it only has planDefinitionPath (no inline definition).
        assert 'query_plan' in deferred_kinds
        # All dify-parity gaps are deferred.
        assert 'tenant_registry' in deferred_kinds
        assert 'rag_index' in deferred_kinds
        assert 'rag_source' in deferred_kinds
        assert 'prompt_package' in deferred_kinds
        assert 'webapp_package' in deferred_kinds
        # sample_data is applied (no storage needed).
        assert 'sample_data' in applied_kinds
        # No artifacts should silently disappear.
        total = len(resp.applied) + len(resp.deferred) + len(resp.failed)
        assert total == len(self._FIXTURE['artifacts'])

    def test_fixture_import_has_partial_status(self) -> None:
        svc = BlueprintImportService()
        resp = svc.import_blueprint('bp.ner-vlm-ocr-entity-extraction', self._FIXTURE)
        # partial: some applied (sample_data), some deferred.
        assert resp.status == 'partial'

    def test_fixture_deferred_reasons_are_non_empty(self) -> None:
        svc = BlueprintImportService()
        resp = svc.import_blueprint('bp.ner-vlm-ocr-entity-extraction', self._FIXTURE)
        for r in resp.deferred:
            assert r.reason is not None and len(r.reason) > 0, (
                f'Deferred artifact {r.ref!r} ({r.kind}) has no reason — never silently skip'
            )
