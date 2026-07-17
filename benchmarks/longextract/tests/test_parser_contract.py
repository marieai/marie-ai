from __future__ import annotations

import json

import marie_longextract.parsers  # noqa: F401
from omegaconf import OmegaConf

from marie.extract.registry import component_registry


def test_longextract_parser_registers() -> None:
    assert component_registry.get_parser('longextract-aggregated') is not None


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
