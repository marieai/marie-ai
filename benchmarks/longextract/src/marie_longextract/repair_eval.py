from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from marie_longextract.agents import application
from marie_longextract.ops.repair import (
    apply_record_patch,
    apply_row_leaf_patches,
)
from marie_longextract.parsers import parse_longextract_aggregated
from omegaconf import OmegaConf

_RAW_ANNOTATOR = 'longextract-unit-extract'
_POLICY_ANNOTATOR = 'longextract-aggregation-policy'


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return value


def _copy_parser_inputs(asset_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    for name in (_RAW_ANNOTATOR, _POLICY_ANNOTATOR, 'tables'):
        source = asset_dir / 'agent-output' / name
        if source.exists():
            shutil.copytree(source, output_dir / 'agent-output' / name)
    frames = asset_dir / 'frames'
    if frames.exists():
        shutil.copytree(frames, output_dir / 'frames')


def _verify_source(asset_dir: Path, page_file: str, expected_sha256: str) -> None:
    source = asset_dir / 'agent-output' / _RAW_ANNOTATOR / page_file
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f'Source artifact changed before proposal apply: {page_file}')


def _aggregate(output_dir: Path) -> None:
    parse_longextract_aggregated(
        None,
        str(output_dir),
        str(output_dir / 'agent-output' / 'longextract-aggregated'),
        OmegaConf.create({}),
    )


def apply_boundary_proposal(
    *,
    asset_dir: Path,
    output_dir: Path,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    if proposal.get('kind') != 'boundary_repair':
        raise ValueError('Expected a boundary repair proposal')
    decision = proposal.get('decision')
    if not isinstance(decision, dict):
        raise ValueError('Boundary proposal decision must be an object')
    page_file = decision.get('page_file')
    if not isinstance(page_file, str) or not page_file:
        raise ValueError('Boundary proposal page_file is required')
    _verify_source(asset_dir, page_file, str(proposal.get('source_sha256') or ''))
    _copy_parser_inputs(asset_dir, output_dir)

    patch = proposal.get('patch')
    if patch is not None:
        if not isinstance(patch, dict):
            raise ValueError('Boundary proposal patch must be an object or null')
        raw_path = output_dir / 'agent-output' / _RAW_ANNOTATOR / page_file
        page_result = _load_json(raw_path)
        policy_dir = output_dir / 'agent-output' / _POLICY_ANNOTATOR
        policy_files = sorted(policy_dir.glob('*.json'))
        if len(policy_files) != 1:
            raise ValueError(f'Expected one aggregation policy in {policy_dir}')
        policy = _load_json(policy_files[0])
        units = policy.get('units')
        if not isinstance(units, dict):
            raise ValueError('Aggregation policy units must be an object')
        repaired = apply_record_patch(
            page_result,
            record_index=int(decision['record_index']),
            is_continuation=bool(patch['is_continuation']),
            unit_name=patch.get('unit_name'),
            active_unit=proposal.get('active_unit'),
            allowed_units=units,
        )
        raw_path.write_text(json.dumps(repaired, indent=2), encoding='utf-8')

    repair_dir = output_dir / 'agent-output' / 'longextract-agent-repair'
    repair_dir.mkdir(parents=True)
    (repair_dir / 'decision.json').write_text(
        json.dumps(decision, indent=2), encoding='utf-8'
    )
    _aggregate(output_dir)
    return {
        'asset_dir': str(asset_dir),
        'output_dir': str(output_dir),
        'active_unit': proposal.get('active_unit'),
        'idempotency_key': proposal.get('idempotency_key'),
        'decision': decision,
    }


def apply_leaf_proposal(
    *,
    asset_dir: Path,
    output_dir: Path,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    if proposal.get('kind') != 'leaf_repair':
        raise ValueError('Expected a leaf repair proposal')
    pages = proposal.get('pages')
    if not isinstance(pages, list) or not all(isinstance(page, dict) for page in pages):
        raise ValueError('Leaf repair proposal pages must be an array of objects')
    for page in pages:
        page_file = page.get('page_file')
        if not isinstance(page_file, str) or not page_file:
            raise ValueError('Leaf proposal page_file is required')
        _verify_source(asset_dir, page_file, str(page.get('source_sha256') or ''))

    _copy_parser_inputs(asset_dir, output_dir)
    repair_dir = output_dir / 'agent-output' / 'longextract-agent-repair'
    repair_dir.mkdir(parents=True)
    for page in pages:
        page_file = page['page_file']
        patches = page.get('patches')
        if not isinstance(patches, list):
            raise ValueError('Leaf proposal patches must be an array')
        raw_path = output_dir / 'agent-output' / _RAW_ANNOTATOR / page_file
        repaired = apply_row_leaf_patches(_load_json(raw_path), patches)
        raw_path.write_text(json.dumps(repaired, indent=2), encoding='utf-8')
        (repair_dir / page_file).write_text(
            json.dumps(page, indent=2), encoding='utf-8'
        )

    _aggregate(output_dir)
    return {
        'asset_dir': str(asset_dir),
        'output_dir': str(output_dir),
        'pages': [int(Path(page['page_file']).stem) for page in pages],
        'fields': proposal.get('fields'),
        'model': proposal.get('model'),
        'idempotency_key': proposal.get('idempotency_key'),
        'patch_count': sum(len(page.get('patches', [])) for page in pages),
        'decisions': pages,
    }


async def run_repair(
    *,
    asset_dir: Path,
    output_dir: Path,
    page_number: int,
    record_index: int,
    api_base: str,
    api_key: str,
    model: str,
    request_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    idempotency_key = f'boundary:{page_number}:{record_index}'
    proposal = await application.propose_boundary_repair(
        asset_dir=asset_dir,
        page_number=page_number,
        record_index=record_index,
        api_base=api_base,
        api_key=api_key,
        model=model,
        idempotency_key=idempotency_key,
        request_timeout_seconds=request_timeout_seconds,
    )
    return apply_boundary_proposal(
        asset_dir=asset_dir,
        output_dir=output_dir,
        proposal=proposal,
    )


async def run_leaf_repair(
    *,
    asset_dir: Path,
    output_dir: Path,
    page_numbers: list[int],
    schema_path: Path,
    api_base: str,
    api_key: str,
    model: str,
    field_names: list[str] | None = None,
    request_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    page_key = ','.join(str(page) for page in sorted(set(page_numbers)))
    field_key = ','.join(sorted(field_names)) if field_names else '*'
    proposal = await application.propose_leaf_repair(
        asset_dir=asset_dir,
        page_numbers=page_numbers,
        schema_path=schema_path,
        api_base=api_base,
        api_key=api_key,
        model=model,
        idempotency_key=f'leaf:{page_key}:{field_key}',
        field_names=field_names,
        request_timeout_seconds=request_timeout_seconds,
    )
    return apply_leaf_proposal(
        asset_dir=asset_dir,
        output_dir=output_dir,
        proposal=proposal,
    )
