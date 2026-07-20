from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from marie_longextract import repair_eval
from marie_longextract.agents import application
from marie_longextract.agents.repair import (
    PageLeafRepairDecision,
    RepairDecision,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


@pytest.fixture
def application_assets(tmp_path: Path) -> tuple[Path, Path]:
    asset_dir = tmp_path / 'source-run'
    raw_dir = asset_dir / 'agent-output' / 'longextract-unit-extract'
    policy_dir = asset_dir / 'agent-output' / 'longextract-aggregation-policy'
    _write_json(
        raw_dir / '00001.json',
        {
            'document_fields': {'title': 'Contract fixture'},
            'records': [
                {
                    'unit_name': 'rows',
                    'source': {'page_index': 0, 'table_index': 0},
                    'continuation': {'is_continuation': False},
                    'rows': [{'label': 'Al pha'}],
                },
                {
                    'unit_name': None,
                    'source': {'page_index': 0, 'table_index': 1},
                    'continuation': {'is_continuation': True},
                    'rows': [{'label': 'Beta'}],
                },
            ],
        },
    )
    (raw_dir / '00001.png_prompt.txt').write_text(
        '\nPage text:\nAlpha\nNOTES\nBeta\n\nPage tables:\n[]',
        encoding='utf-8',
    )
    _write_json(
        policy_dir / '00001.json',
        {
            'units': {
                'rows': {'carry_fields': [], 'sequence_fields': []},
                'notes': {'carry_fields': [], 'sequence_fields': []},
            }
        },
    )
    schema_path = tmp_path / 'schema.json'
    field_schema = {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'label': {
                    'type': 'string',
                    'description': 'The exact printed label.',
                }
            },
        },
    }
    _write_json(
        schema_path,
        {
            'type': 'object',
            'properties': {'rows': field_schema, 'notes': field_schema},
        },
    )
    return asset_dir, schema_path


def _boundary_decision(marker: str) -> RepairDecision:
    return RepairDecision.model_validate(
        {
            'page_file': '00001.json',
            'record_index': 1,
            'boundary': {
                'kind': 'new_schema_unit',
                'is_continuation': False,
                'unit_name': 'notes',
                'new_unit_marker': marker,
            },
            'sequence_evidence': ['The second record follows the first record.'],
            'schema_evidence': ['NOTES starts the notes unit.'],
            'evidence': [marker],
            'rationale': 'The printed heading starts a new schema unit.',
        }
    )


def _leaf_decision(replacement: str) -> PageLeafRepairDecision:
    return PageLeafRepairDecision.model_validate(
        {
            'page_file': '00001.json',
            'reviews': [
                {
                    'record_index': 0,
                    'row_index': 0,
                    'field_name': 'label',
                    'action': 'use_source_candidate',
                    'evidence_source': 'current_page_text',
                    'evidence_line': 1,
                    'evidence_quote': replacement,
                    'rationale': 'The prepared page shows the unbroken label.',
                },
                {
                    'record_index': 1,
                    'row_index': 0,
                    'field_name': 'label',
                    'action': 'use_source_candidate',
                    'evidence_source': 'current_page_text',
                    'evidence_line': 3,
                    'evidence_quote': 'Beta',
                    'rationale': 'The prepared page supports the current label.',
                },
            ],
            'rationale': 'Repair the exact string leaf.',
        }
    )


@pytest.mark.asyncio
async def test_boundary_repair_contract_validates_proposes_applies_and_aggregates(
    application_assets: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_dir, _schema_path = application_assets
    output_dir = tmp_path / 'boundary-output'
    source_path = asset_dir / 'agent-output' / 'longextract-unit-extract' / '00001.json'
    source_page = source_path.read_bytes()
    events: list[str] = []
    feedback: list[str | None] = []
    decisions = iter((_boundary_decision('MISSING'), _boundary_decision('NOTES')))

    async def run_agent(**kwargs: Any) -> RepairDecision:
        events.append('model')
        feedback.append(kwargs['validation_feedback'])
        return next(decisions)

    original_proposal = application.record_patch_from_decision
    original_apply = repair_eval.apply_record_patch
    original_aggregate = repair_eval._aggregate

    def propose(
        decision: RepairDecision,
        current_record: dict[str, object],
    ) -> Any:
        events.append('proposal')
        return original_proposal(decision, current_record)

    def apply(page_result: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        events.append('apply')
        return original_apply(page_result, **kwargs)

    def aggregate(output_dir: Path) -> None:
        events.append('aggregate')
        original_aggregate(output_dir)

    monkeypatch.setattr(application, '_run_agent', run_agent)
    monkeypatch.setattr(application, 'record_patch_from_decision', propose)
    monkeypatch.setattr(repair_eval, 'apply_record_patch', apply)
    monkeypatch.setattr(repair_eval, '_aggregate', aggregate)

    report = await repair_eval.run_repair(
        asset_dir=asset_dir,
        output_dir=output_dir,
        page_number=1,
        record_index=1,
        api_base='http://model.test',
        api_key='test-only',
        model='contract-model',
    )

    assert events == ['model', 'model', 'proposal', 'apply', 'aggregate']
    assert feedback == [
        None,
        'new_unit_marker is not grounded in the current page text',
    ]
    assert report['decision']['action'] == 'patch'
    assert report['decision']['boundary']['unit_name'] == 'notes'
    assert json.loads(
        (output_dir / 'parsed-result' / 'longextract-result.json').read_text()
    ) == {
        'title': 'Contract fixture',
        'rows': [{'label': 'Al pha'}],
        'notes': [{'label': 'Beta'}],
    }
    assert source_path.read_bytes() == source_page


@pytest.mark.asyncio
async def test_leaf_repair_contract_retries_validates_votes_applies_and_aggregates(
    application_assets: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_dir, schema_path = application_assets
    output_dir = tmp_path / 'leaf-output'
    source_path = asset_dir / 'agent-output' / 'longextract-unit-extract' / '00001.json'
    source_page = source_path.read_bytes()
    events: list[str] = []
    feedback: list[str | None] = []
    decisions = iter(
        (
            _leaf_decision('Invented'),
            _leaf_decision('Alpha'),
            _leaf_decision('Alpha'),
            _leaf_decision('Alpha'),
        )
    )

    async def run_leaf_agent(**kwargs: Any) -> PageLeafRepairDecision:
        events.append('model')
        feedback.append(kwargs['validation_feedback'])
        return next(decisions)

    original_consensus = application.select_leaf_repair_consensus
    original_apply = repair_eval.apply_row_leaf_patches
    original_aggregate = repair_eval._aggregate

    def select_consensus(**kwargs: Any) -> PageLeafRepairDecision:
        events.append('proposal')
        return original_consensus(**kwargs)

    def apply(
        page_result: dict[str, Any],
        patches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        events.append('apply')
        return original_apply(page_result, patches)

    def aggregate(output_dir: Path) -> None:
        events.append('aggregate')
        original_aggregate(output_dir)

    monkeypatch.setattr(application, '_run_leaf_agent', run_leaf_agent)
    monkeypatch.setattr(application, 'select_leaf_repair_consensus', select_consensus)
    monkeypatch.setattr(repair_eval, 'apply_row_leaf_patches', apply)
    monkeypatch.setattr(repair_eval, '_aggregate', aggregate)

    report = await repair_eval.run_leaf_repair(
        asset_dir=asset_dir,
        output_dir=output_dir,
        page_numbers=[1],
        schema_path=schema_path,
        api_base='http://model.test',
        api_key='test-only',
        model='contract-model',
        field_names=['label'],
    )

    assert events == [
        'model',
        'model',
        'model',
        'model',
        'proposal',
        'apply',
        'aggregate',
    ]
    assert feedback[0] is None
    assert feedback[1] == (
        "Evidence quote for 'label' is not an exact line candidate at "
        'current_page_text L0001'
    )
    assert feedback[2:] == [None, None]
    assert report['patch_count'] == 1
    assert report['decisions'][0]['patches'][0]['replacement_value'] == 'Alpha'
    assert json.loads(
        (output_dir / 'parsed-result' / 'longextract-result.json').read_text()
    ) == {
        'title': 'Contract fixture',
        'rows': [{'label': 'Alpha'}, {'label': 'Beta'}],
    }
    assert source_path.read_bytes() == source_page


def test_runtime_uses_public_agent_api_and_benchmark_wrappers_own_grading() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_source = (
        root / 'src' / 'marie_longextract' / 'agents' / 'application.py'
    ).read_text(encoding='utf-8')
    host_source = (root / 'src' / 'marie_longextract' / 'repair_eval.py').read_text(
        encoding='utf-8',
    )
    assert 'from marie.agent.agents import ReactAgent' in runtime_source
    assert 'from marie.agent.llm import OpenAICompatibleWrapper' in runtime_source
    assert 'from marie.agent.messages import ContentItem, Message' in runtime_source
    assert 'longextract_bench.grading' not in runtime_source
    assert 'ReactAgent' not in host_source

    for name in ('evaluate-agent-leaves.py', 'evaluate-agent-repair.py'):
        wrapper_source = (root / 'tools' / name).read_text(encoding='utf-8')
        assert 'from longextract_bench.grading import grade' in wrapper_source
        assert '--ground-truth' in wrapper_source
