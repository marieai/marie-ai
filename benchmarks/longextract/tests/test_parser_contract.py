from __future__ import annotations

import json
import os
import shutil

import marie_longextract.parsers  # noqa: F401
from omegaconf import OmegaConf

from marie.extract.registry import component_registry


def test_longextract_parser_registers() -> None:
    assert component_registry.get_parser('longextract-aggregated') is not None


def test_parser_rejects_a_schema_invalid_refinement_without_changing_raw_passes(
    tmp_path,
) -> None:
    raw = tmp_path / 'agent-output/longextract-unit-extract'
    run = tmp_path / 'agent-output/.longextract-unit-extract-refine/runs/current'
    policy = tmp_path / 'agent-output/longextract-aggregation-policy'
    debug = tmp_path / 'debug'
    for directory in (raw, run / 'pass_0', run / 'pass_1', policy, debug):
        directory.mkdir(parents=True)

    def page(region, value):
        return {
            'document_fields': {'title': 'Example'},
            'records': (
                []
                if value is None
                else [
                    {
                        'unit_name': 'rows',
                        'continuation': {'is_continuation': False},
                        'rows': [{'region': region, 'value': value}],
                    }
                ]
            ),
        }

    initial = [page(None, None), page('AREA A', 1)]
    refined = [page(None, 999), page('AREA A', 2)]
    for index in range(2):
        name = f'{index + 1:05d}.json'
        (run / 'pass_0' / name).write_text(json.dumps(initial[index]))
        (run / 'pass_1' / name).write_text(json.dumps(refined[index]))
        (raw / name).write_text(json.dumps(refined[index]))
    schema_units = [
        {
            'unit_name': 'rows',
            'unit_kind': 'array',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'required': ['region', 'value'],
                    'properties': {
                        'region': {'type': 'string'},
                        'value': {'type': 'integer'},
                    },
                },
            },
        }
    ]
    (debug / 'longextract-unit-extract_context.json').write_text(
        json.dumps(
            {
                'units': [
                    {'variables': {'SCHEMA_UNITS_JSON': json.dumps(schema_units)}}
                ],
            }
        )
    )
    (policy / '00001.json').write_text(
        json.dumps({'units': {'rows': {'carry_fields': [], 'sequence_fields': []}}})
    )
    stale = run.parent / 'stale'
    shutil.copytree(run, stale)
    (stale / 'pass_0/00001.json').write_text(json.dumps(page('STALE', 123)))
    os.utime(
        stale, ns=(run.stat().st_atime_ns, run.stat().st_mtime_ns + 10_000_000_000)
    )
    (raw / '_SUCCESS.yaml').write_text('run_id: current\npromoted_pass: pass_1\n')

    component_registry.get_parser('longextract-aggregated')(
        None,
        str(tmp_path),
        str(tmp_path / 'agent-output/longextract-aggregated'),
        OmegaConf.create({}),
    )

    result = json.loads(
        (tmp_path / 'parsed-result/longextract-result.json').read_text()
    )
    assert result == {'title': 'Example', 'rows': [{'region': 'AREA A', 'value': 2}]}
    assert json.loads((raw / '00001.json').read_text()) == refined[0]
    audit = json.loads((debug / 'longextract-refinement-selection.json').read_text())
    assert audit['fallbacks'][0]['filename'] == '00001.json'
    assert audit['fallbacks'][0]['candidate_error_count'] == 1


def test_longextract_parser_writes_aggregated_result_and_trace(tmp_path) -> None:
    raw_dir = tmp_path / 'agent-output' / 'longextract-unit-extract'
    output_dir = tmp_path / 'agent-output' / 'longextract-aggregated'
    policy_dir = tmp_path / 'agent-output' / 'longextract-aggregation-policy'
    raw_dir.mkdir(parents=True)
    policy_dir.mkdir(parents=True)
    raw_result = {
        'document_fields': {'title': 'Example'},
        'records': [
            {
                'unit_name': 'rows',
                'source': {'page_index': 7, 'table_index': 0},
                'continuation': {'is_continuation': False},
                'rows': [{'row_order': 1, 'value': 'A'}],
            }
        ],
    }
    (raw_dir / '00001.json').write_text(json.dumps(raw_result), encoding='utf-8')
    (policy_dir / '00001.json').write_text(
        json.dumps(
            {
                'units': {
                    'rows': {
                        'carry_fields': [],
                        'sequence_fields': ['row_order'],
                    }
                }
            }
        ),
        encoding='utf-8',
    )

    parser = component_registry.get_parser('longextract-aggregated')
    assert parser is not None
    parser(None, str(tmp_path), str(output_dir), OmegaConf.create({}))

    expected = {'title': 'Example', 'rows': [{'row_order': 1, 'value': 'A'}]}
    assert json.loads((output_dir / '00001.json').read_text()) == expected
    assert (
        json.loads((tmp_path / 'parsed-result' / 'longextract-result.json').read_text())
        == expected
    )
    assert '# LongExtract Aggregation Trace' in (output_dir / 'trace.md').read_text()
