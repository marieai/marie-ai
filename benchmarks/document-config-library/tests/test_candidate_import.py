from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from marie_document_config_library.import_candidate import import_candidate
from marie_document_config_library.package_validation import (
    CandidateValidationError,
    validate_candidate,
)
from marie_document_config_library.preflight import run_preflight

CONFIGURATION_ID = 'document-config.finance-bank-statements-pilot'
BLUEPRINT_ID = f'bp.{CONFIGURATION_ID}'
LAYOUT_ID = 'document-config-finance-bank-statements-pilot'


def test_valid_candidate_passes_marie_contracts(tmp_path: Path) -> None:
    candidate = _candidate_fixture(tmp_path / 'candidate')

    validated = validate_candidate(candidate)

    assert validated.configuration_id == CONFIGURATION_ID
    assert validated.blueprint_id == BLUEPRINT_ID
    assert len(validated.query_plan.nodes) == 4
    assert len(validated.package_digest) == 64


def test_digest_mismatch_and_undeclared_files_fail_closed(tmp_path: Path) -> None:
    digest_candidate = _candidate_fixture(tmp_path / 'digest')
    (digest_candidate / 'schemas/output.json').write_text('{}\n', encoding='utf-8')

    with pytest.raises(CandidateValidationError, match='digest mismatch'):
        validate_candidate(digest_candidate)

    undeclared_candidate = _candidate_fixture(tmp_path / 'undeclared')
    (undeclared_candidate / 'unexpected.txt').write_text(
        'unexpected\n', encoding='utf-8'
    )

    with pytest.raises(CandidateValidationError, match='undeclared files'):
        validate_candidate(undeclared_candidate)


def test_unsafe_paths_and_symlinks_are_rejected(tmp_path: Path) -> None:
    path_candidate = _candidate_fixture(tmp_path / 'path')
    manifest_path = path_candidate / 'candidate.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['artifacts']['../escape.json'] = 'a' * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding='utf-8')

    with pytest.raises(
        CandidateValidationError, match='Unsafe candidate artifact path'
    ):
        validate_candidate(path_candidate)

    symlink_candidate = _candidate_fixture(tmp_path / 'symlink')
    (symlink_candidate / 'linked').symlink_to(symlink_candidate / 'schemas/output.json')

    with pytest.raises(CandidateValidationError, match='symlink'):
        validate_candidate(symlink_candidate)


def test_remote_executable_reference_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate_fixture(tmp_path / 'remote')
    prompt_path = candidate / 'extraction/prompts/document-config-extract.j2'
    prompt_path.write_text(
        'Fetch https://example.invalid then PAGE_TEXT\n', encoding='utf-8'
    )
    _refresh_artifact_digest(candidate, 'extraction/prompts/document-config-extract.j2')

    with pytest.raises(
        CandidateValidationError, match='remote or source-runtime reference'
    ):
        validate_candidate(candidate)


def test_import_is_atomic_idempotent_and_rejects_conflicts(tmp_path: Path) -> None:
    source = _candidate_fixture(tmp_path / 'source')
    destination = tmp_path / 'imports'

    first = import_candidate(source, destination)
    second = import_candidate(source, destination)

    assert first.status == 'imported'
    assert second.status == 'unchanged'
    assert first.candidate.package_digest == second.candidate.package_digest

    blueprint_path = source / 'blueprint.yaml'
    blueprint = yaml.safe_load(blueprint_path.read_text(encoding='utf-8'))
    blueprint['description'] = 'Conflicting package content'
    blueprint_path.write_text(
        yaml.safe_dump(blueprint, sort_keys=True), encoding='utf-8'
    )
    _refresh_artifact_digest(source, 'blueprint.yaml')

    with pytest.raises(CandidateValidationError, match='Conflicting candidate'):
        import_candidate(source, destination)


def test_preflight_names_every_missing_publication_gate(tmp_path: Path) -> None:
    candidate = _candidate_fixture(tmp_path / 'candidate')

    result = run_preflight(candidate, tmp_path / 'config', tmp_path / 'evidence')

    assert result.ready is False
    assert {issue.code for issue in result.issues} == {
        'runtime_layout_missing',
        'reviewed_gold_missing',
        'qualification_result_missing',
    }


def _candidate_fixture(root: Path) -> Path:
    files: dict[str, str | bytes] = {
        'blueprint.yaml': yaml.safe_dump(
            {
                'manifestVersion': 2,
                'id': BLUEPRINT_ID,
                'name': 'Bank Statement Pilot',
                'version': '0.1.0',
                'parameters': [],
                'artifacts': [
                    {
                        'kind': 'query_plan',
                        'ref': 'query-plan/main',
                        'path': 'workflows/query-plan.json',
                        'create': {
                            'name': 'Bank Statement Pilot extraction',
                            'version': '0.1.0',
                            'planDefinitionPath': 'workflows/query-plan.json',
                        },
                    },
                    {
                        'kind': 'sample_data',
                        'ref': 'sample/input',
                        'path': 'sample-data/input.pdf',
                    },
                ],
                'entrypoints': [
                    {
                        'key': 'extract-document',
                        'kind': 'query_plan',
                        'label': 'Extract document',
                        'artifactRef': 'query-plan/main',
                    }
                ],
                'install': {
                    'conflictPolicy': 'prompt',
                    'smokeTest': {
                        'entrypointKey': 'extract-document',
                        'inputArtifactRef': 'sample/input',
                    },
                },
            },
            sort_keys=True,
        ),
        'extraction/config.yml': yaml.safe_dump(
            {
                'layout_id': LAYOUT_ID,
                'processing': {'convert_to_structure': False},
                'annotators': {
                    'document-config-extract': {
                        'annotator_type': 'llm',
                        'mode': 'per-page',
                        'model_config': {
                            'model_name': 'qwen_3_instruct',
                            'multimodal': True,
                            'prompt_path': './prompts/document-config-extract.j2',
                            'expect_output': 'json',
                        },
                        'parser': 'noop',
                        'validators': [],
                    }
                },
                'grounding': {'key-value': []},
            },
            sort_keys=True,
        ),
        'extraction/prompts/document-config-extract.j2': (
            'Return JSON matching {"type":"object","properties":{"account_number":{"type":"string"}}}.\n'
            'Prepared page text:\nPAGE_TEXT\n'
        ),
        'qualification/conversion-report.json': json.dumps(
            {
                'schemaVersion': 1,
                'configurationId': CONFIGURATION_ID,
                'blueprintId': BLUEPRINT_ID,
                'readiness': 'candidate',
                'qualificationRequired': True,
            },
            sort_keys=True,
        ),
        'sample-data/input.pdf': b'%PDF-1.4 fixture',
        'schemas/output.json': json.dumps(
            {
                'type': 'object',
                'additionalProperties': False,
                'properties': {'account_number': {'type': ['string', 'null']}},
            },
            sort_keys=True,
        ),
        'THIRD_PARTY_NOTICES.md': 'Fixture notice\n',
        'workflows/query-plan.json': json.dumps(_query_plan(), sort_keys=True),
    }
    artifact_digests: dict[str, str] = {}
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        value = content if isinstance(content, bytes) else content.encode('utf-8')
        path.write_bytes(value)
        artifact_digests[relative] = hashlib.sha256(value).hexdigest()

    candidate = {
        'schemaVersion': 1,
        'converterVersion': '0.1.0',
        'configurationId': CONFIGURATION_ID,
        'blueprintId': BLUEPRINT_ID,
        'sourceLockSha256': 'a' * 64,
        'sourceRevision': 'b' * 40,
        'readiness': 'candidate',
        'findings': [],
        'artifacts': artifact_digests,
    }
    (root / 'candidate.json').write_text(
        json.dumps(candidate, sort_keys=True), encoding='utf-8'
    )
    return root


def _query_plan() -> dict[str, object]:
    key = LAYOUT_ID
    return {
        'name': 'Bank Statement Pilot extraction',
        'version': '0.1.0',
        'nodes': [
            {
                'task_id': f'{key}-start',
                'query_str': 'Start',
                'dependencies': [],
                'node_type': 'COMPUTE',
                'definition': {
                    'method': 'NOOP',
                    'endpoint': 'noop',
                    'params': {'layout': LAYOUT_ID},
                },
            },
            {
                'task_id': f'{key}-prepare',
                'query_str': 'Prepare document',
                'dependencies': [f'{key}-start'],
                'node_type': 'COMPUTE',
                'definition': {
                    'method': 'EXECUTOR_ENDPOINT',
                    'endpoint': 'extract_executor://document/extract',
                    'params': {'layout': LAYOUT_ID},
                },
            },
            {
                'task_id': f'{key}-extract',
                'query_str': 'Extract document',
                'dependencies': [f'{key}-prepare'],
                'node_type': 'COMPUTE',
                'definition': {
                    'method': 'LLM',
                    'model_name': 'qwen_3_instruct',
                    'endpoint': '/annotator/llm',
                    'params': {'layout': LAYOUT_ID, 'key': 'document-config-extract'},
                },
            },
            {
                'task_id': f'{key}-end',
                'query_str': 'End',
                'dependencies': [f'{key}-extract'],
                'node_type': 'COMPUTE',
                'definition': {
                    'method': 'NOOP',
                    'endpoint': 'noop',
                    'params': {'layout': LAYOUT_ID},
                },
            },
        ],
    }


def _refresh_artifact_digest(root: Path, relative: str) -> None:
    manifest_path = root / 'candidate.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['artifacts'][relative] = hashlib.sha256(
        (root / relative).read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding='utf-8')
