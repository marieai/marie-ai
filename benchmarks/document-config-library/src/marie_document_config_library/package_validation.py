"""Fail-closed validation for neutral document configuration candidates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from marie.query_planner import QueryPlan
from marie.query_planner.mapper import JobMetadata
from marie_document_config_library.paths import (
    CandidatePathError,
    package_files,
    safe_artifact_path,
)

_DIGEST = re.compile(r'^[a-f0-9]{64}$')
_CONFIGURATION_ID = re.compile(r'^document-config\.[a-z0-9][a-z0-9.-]*$')
_BLUEPRINT_ID = re.compile(r'^bp\.document-config\.[a-z0-9][a-z0-9.-]*$')
_REMOTE_OR_SOURCE_RUNTIME = re.compile(
    r'https?://|raw\.githubusercontent|sensible-hq|sensible-configuration-library|senseml',
    re.IGNORECASE,
)
_REQUIRED_ARTIFACTS = {
    'blueprint.yaml',
    'extraction/config.yml',
    'extraction/prompts/document-config-extract.j2',
    'qualification/conversion-report.json',
    'sample-data/input.pdf',
    'schemas/output.json',
    'THIRD_PARTY_NOTICES.md',
    'workflows/query-plan.json',
}
_SCAN_SUFFIXES = {'.j2', '.json', '.yaml', '.yml'}


class CandidateValidationError(ValueError):
    """Raised when a candidate package violates the import contract."""


@dataclass(frozen=True)
class ValidatedCandidate:
    """Validated candidate identity and package evidence."""

    root: Path
    configuration_id: str
    blueprint_id: str
    source_lock_sha256: str
    package_digest: str
    artifact_digests: dict[str, str]
    query_plan: QueryPlan
    blueprint: dict[str, Any]


def validate_candidate(root: str | Path) -> ValidatedCandidate:
    """Validate a Studio-generated candidate without executing its contents.

    Args:
        root: Candidate package directory.

    Returns:
        Validated candidate metadata.

    Raises:
        CandidateValidationError: If any package, digest, or Marie contract fails.
    """
    candidate_root = Path(root).expanduser()
    try:
        files = package_files(candidate_root)
    except CandidatePathError as error:
        raise CandidateValidationError(str(error)) from error

    candidate_path = candidate_root / 'candidate.json'
    if 'candidate.json' not in files:
        raise CandidateValidationError('Candidate package is missing candidate.json')
    manifest = _read_json(candidate_path, 'candidate manifest')

    configuration_id = manifest.get('configurationId')
    blueprint_id = manifest.get('blueprintId')
    source_lock = manifest.get('sourceLockSha256')
    artifact_digests = manifest.get('artifacts')
    if manifest.get('schemaVersion') != 1:
        raise CandidateValidationError('Unsupported candidate schema version')
    if manifest.get('readiness') != 'candidate':
        raise CandidateValidationError('Candidate readiness must be candidate')
    if not isinstance(configuration_id, str) or not _CONFIGURATION_ID.fullmatch(
        configuration_id
    ):
        raise CandidateValidationError('Invalid Marie configuration ID')
    if not isinstance(blueprint_id, str) or not _BLUEPRINT_ID.fullmatch(blueprint_id):
        raise CandidateValidationError('Invalid Marie Blueprint ID')
    if blueprint_id != f'bp.{configuration_id}':
        raise CandidateValidationError('Blueprint ID does not match configuration ID')
    if not isinstance(source_lock, str) or not _DIGEST.fullmatch(source_lock):
        raise CandidateValidationError('Invalid or missing source-lock digest')
    if not isinstance(artifact_digests, dict) or not artifact_digests:
        raise CandidateValidationError('Candidate artifact digest map is missing')

    normalized_digests: dict[str, str] = {}
    for raw_path, raw_digest in artifact_digests.items():
        if not isinstance(raw_path, str) or not isinstance(raw_digest, str):
            raise CandidateValidationError(
                'Candidate artifact entries must map paths to SHA-256 digests'
            )
        if not _DIGEST.fullmatch(raw_digest):
            raise CandidateValidationError(f'Invalid SHA-256 digest for {raw_path}')
        try:
            artifact_path = safe_artifact_path(candidate_root, raw_path)
        except CandidatePathError as error:
            raise CandidateValidationError(str(error)) from error
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise CandidateValidationError(
                f'Declared candidate artifact is missing: {raw_path}'
            )
        actual_digest = _sha256(artifact_path.read_bytes())
        if actual_digest != raw_digest:
            raise CandidateValidationError(
                f'Candidate artifact digest mismatch: {raw_path}'
            )
        normalized_digests[raw_path] = raw_digest

    missing_required = sorted(_REQUIRED_ARTIFACTS - normalized_digests.keys())
    if missing_required:
        raise CandidateValidationError(
            f'Candidate is missing required artifacts: {", ".join(missing_required)}'
        )

    expected_files = {'candidate.json', *normalized_digests.keys()}
    undeclared = sorted(files - expected_files)
    missing = sorted(expected_files - files)
    if undeclared:
        raise CandidateValidationError(
            f'Candidate contains undeclared files: {", ".join(undeclared)}'
        )
    if missing:
        raise CandidateValidationError(
            f'Candidate is missing declared files: {", ".join(missing)}'
        )

    _reject_remote_executable_references(candidate_root, normalized_digests)
    blueprint = _validate_blueprint(candidate_root, blueprint_id)
    query_plan = _validate_query_plan(candidate_root, blueprint)
    _validate_extraction_config(candidate_root, query_plan)
    _validate_output_schema(candidate_root)

    package_digest = _package_digest(candidate_path.read_bytes(), normalized_digests)
    return ValidatedCandidate(
        root=candidate_root.resolve(),
        configuration_id=configuration_id,
        blueprint_id=blueprint_id,
        source_lock_sha256=source_lock,
        package_digest=package_digest,
        artifact_digests=normalized_digests,
        query_plan=query_plan,
        blueprint=blueprint,
    )


def _validate_blueprint(root: Path, blueprint_id: str) -> dict[str, Any]:
    raw = yaml.safe_load((root / 'blueprint.yaml').read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise CandidateValidationError('Blueprint manifest must be a mapping')
    if raw.get('manifestVersion') != 2 or raw.get('id') != blueprint_id:
        raise CandidateValidationError(
            'Blueprint manifest identity does not match candidate'
        )
    artifacts = raw.get('artifacts')
    if not isinstance(artifacts, list):
        raise CandidateValidationError('Blueprint artifacts must be a list')
    query_artifacts = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get('kind') == 'query_plan'
    ]
    if len(query_artifacts) != 1:
        raise CandidateValidationError(
            'Blueprint must declare exactly one query-plan artifact'
        )
    create = query_artifacts[0].get('create')
    if (
        not isinstance(create, dict)
        or create.get('planDefinitionPath') != 'workflows/query-plan.json'
    ):
        raise CandidateValidationError(
            'Blueprint query-plan artifact is not installable by Studio'
        )
    return raw


def _validate_query_plan(root: Path, blueprint: dict[str, Any]) -> QueryPlan:
    raw = _read_json(root / 'workflows/query-plan.json', 'query plan')
    try:
        plan = QueryPlan(**raw)
    except Exception as error:
        raise CandidateValidationError(f'Invalid Marie query plan: {error}') from error

    if len(plan.nodes) < 4:
        raise CandidateValidationError(
            'Query plan must include preparation and LLM annotation nodes'
        )
    layout = _query_plan_layout(plan)
    routes = [JobMetadata.from_task(node, layout).metadata.on for node in plan.nodes]
    if 'extract_executor://document/extract' not in routes:
        raise CandidateValidationError(
            'Query plan does not use the document extraction route'
        )
    if not any(route.endswith('://annotator/llm') for route in routes):
        raise CandidateValidationError(
            'Query plan does not use the LLM annotator route'
        )

    llm_nodes = [node for node in plan.nodes if node.definition.method == 'LLM']
    if len(llm_nodes) != 1:
        raise CandidateValidationError(
            'Query plan must declare exactly one LLM annotation node'
        )
    llm_node = llm_nodes[0]
    if llm_node.definition.params.get('key') != 'document-config-extract':
        raise CandidateValidationError(
            'Query plan LLM node does not target the candidate annotator'
        )

    entrypoints = blueprint.get('entrypoints')
    if not isinstance(entrypoints, list) or not entrypoints:
        raise CandidateValidationError('Blueprint must declare a query-plan entrypoint')
    return plan


def _query_plan_layout(plan: QueryPlan) -> str:
    layouts = {
        node.definition.params.get('layout')
        for node in plan.nodes
        if isinstance(node.definition.params, dict)
        and node.definition.params.get('layout')
    }
    if len(layouts) != 1:
        raise CandidateValidationError(
            'All query-plan nodes must use one non-empty layout'
        )
    layout = next(iter(layouts))
    if not isinstance(layout, str):
        raise CandidateValidationError('Query-plan layout must be a string')
    return layout


def _validate_extraction_config(root: Path, plan: QueryPlan) -> None:
    raw = yaml.safe_load((root / 'extraction/config.yml').read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise CandidateValidationError('Extraction config must be a mapping')
    if raw.get('layout_id') != _query_plan_layout(plan):
        raise CandidateValidationError(
            'Extraction config layout does not match query plan'
        )
    annotators = raw.get('annotators')
    annotator = (
        annotators.get('document-config-extract')
        if isinstance(annotators, dict)
        else None
    )
    model_config = (
        annotator.get('model_config') if isinstance(annotator, dict) else None
    )
    if (
        not isinstance(model_config, dict)
        or model_config.get('model_name') != 'qwen_3_instruct'
    ):
        raise CandidateValidationError(
            'Extraction config does not select the qualified Marie model role'
        )
    prompt_path = model_config.get('prompt_path')
    if prompt_path != './prompts/document-config-extract.j2':
        raise CandidateValidationError(
            'Extraction config uses an unexpected prompt path'
        )
    prompt = (root / 'extraction/prompts/document-config-extract.j2').read_text(
        encoding='utf-8'
    )
    if (
        'PAGE_TEXT' not in prompt
        or '{{ output_schema }}' in prompt
        or '{{ page_text }}' in prompt
    ):
        raise CandidateValidationError(
            'Extraction prompt does not use Marie runtime variables'
        )


def _validate_output_schema(root: Path) -> None:
    schema = _read_json(root / 'schemas/output.json', 'output schema')
    if schema.get('type') != 'object' or not isinstance(schema.get('properties'), dict):
        raise CandidateValidationError(
            'Candidate output schema must describe an object'
        )


def _reject_remote_executable_references(root: Path, artifacts: dict[str, str]) -> None:
    for artifact in artifacts:
        path = Path(artifact)
        if (
            artifact == 'THIRD_PARTY_NOTICES.md'
            or path.suffix.lower() not in _SCAN_SUFFIXES
        ):
            continue
        text = (root / artifact).read_text(encoding='utf-8')
        if _REMOTE_OR_SOURCE_RUNTIME.search(text):
            raise CandidateValidationError(
                f'Executable candidate artifact contains a remote or source-runtime reference: {artifact}'
            )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateValidationError(f'Invalid {label}: {error}') from error
    if not isinstance(raw, dict):
        raise CandidateValidationError(f'{label.capitalize()} must be an object')
    return raw


def _package_digest(candidate_bytes: bytes, artifacts: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(candidate_bytes)
    for path, artifact_digest in sorted(artifacts.items()):
        digest.update(path.encode('utf-8'))
        digest.update(b'\0')
        digest.update(artifact_digest.encode('ascii'))
        digest.update(b'\0')
    return digest.hexdigest()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
