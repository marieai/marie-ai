from __future__ import annotations

from pathlib import Path

import pytest
from marie_longextract.models import PartialExtraction
from marie_longextract.ops.aggregation import aggregate_page_results
from marie_longextract.ops.schema import build_extraction_units
from marie_longextract.ops.stitch import stitch_partials
from marie_longextract.ops.verify import verify_result


def _schema() -> dict:
    return {
        'type': 'object',
        'required': ['claim_number', 'service_lines'],
        'properties': {
            'claim_number': {'type': ['string', 'null']},
            'payer': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
            'service_lines': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'code': {'type': 'string'},
                        'amount': {'type': 'string'},
                    },
                },
            },
        },
    }


def test_schema_decomposition_builds_processing_unit_contracts() -> None:
    units = build_extraction_units(_schema())
    assert [(unit['unit_name'], unit['unit_kind']) for unit in units] == [
        ('document_fields', 'object'),
        ('service_lines', 'array'),
    ]


def test_stitch_dedupes_rows_and_sets_attempted_nullable_fields() -> None:
    partials = [
        PartialExtraction(
            unit_name='document_fields',
            page_index=0,
            rows=[],
            scalars={'claim_number': 'ABC'},
            source_uri='memory://document',
        ),
        PartialExtraction(
            unit_name='service_lines',
            page_index=0,
            rows=[{'code': ' 99213 ', 'amount': '10.00'}],
            scalars={},
            source_uri='memory://page-1',
        ),
        PartialExtraction(
            unit_name='service_lines',
            page_index=1,
            rows=[{'amount': '10.00', 'code': '99213'}],
            scalars={},
            source_uri='memory://page-2',
        ),
    ]
    result = stitch_partials(partials, _schema())
    assert result == {
        'claim_number': 'ABC',
        'payer': None,
        'service_lines': [{'code': ' 99213 ', 'amount': '10.00'}],
    }


def test_page_aggregation_applies_explicit_continuation_policy() -> None:
    result, trace = aggregate_page_results(
        [
            (
                '00001.json',
                {
                    'document_fields': {'report_name': 'Example'},
                    'records': [
                        {
                            'unit_name': 'rows',
                            'source': {'page_index': 99, 'table_index': 0},
                            'continuation': {'is_continuation': False},
                            'rows': [
                                {
                                    'row_order': 1,
                                    'heading': 'GROUP A',
                                    'value': '(X)',
                                }
                            ],
                        }
                    ],
                },
            ),
            (
                '00002.json',
                {
                    'document_fields': {'report_name': None},
                    'records': [
                        {
                            'unit_name': None,
                            'source': {'page_index': 99, 'table_index': 0},
                            'continuation': {'is_continuation': True},
                            'rows': [{'row_order': 1, 'heading': None, 'value': 'X'}],
                        }
                    ],
                },
            ),
        ],
        {
            'units': {
                'rows': {
                    'carry_fields': ['heading'],
                    'sequence_fields': ['row_order'],
                }
            }
        },
    )

    assert result == {
        'report_name': 'Example',
        'rows': [
            {'row_order': 1, 'heading': 'GROUP A', 'value': '(X)'},
            {'row_order': 2, 'heading': 'GROUP A', 'value': 'X'},
        ],
    }
    assert [entry['action'] for entry in trace] == [
        'NEW_PARENT',
        'MERGE',
        'SUMMARY',
    ]
    assert trace[1]['source']['page_index'] == 1


def test_page_aggregation_rejects_contradictory_continuation_unit() -> None:
    with pytest.raises(ValueError, match='contradicts active unit'):
        aggregate_page_results(
            [
                (
                    '00001.json',
                    {
                        'document_fields': {},
                        'records': [
                            {
                                'unit_name': 'first',
                                'source': {},
                                'continuation': {'is_continuation': False},
                                'rows': [],
                            },
                            {
                                'unit_name': 'second',
                                'source': {},
                                'continuation': {'is_continuation': True},
                                'rows': [],
                            },
                        ],
                    },
                )
            ],
            {
                'units': {
                    'first': {'carry_fields': [], 'sequence_fields': []},
                    'second': {'carry_fields': [], 'sequence_fields': []},
                }
            },
        )


def test_page_aggregation_skips_empty_anonymous_orphan_continuation() -> None:
    result, trace = aggregate_page_results(
        [
            (
                '00006.json',
                {
                    'document_fields': {'report_name': 'Example'},
                    'records': [
                        {
                            'unit_name': None,
                            'source': {'page_index': 99, 'table_index': 0},
                            'continuation': {'is_continuation': True},
                            'rows': [],
                        },
                        {
                            'unit_name': 'rows',
                            'continuation': {'is_continuation': False},
                            'rows': [{'value': 'A'}],
                        },
                    ],
                },
            )
        ],
        {'units': {'rows': {'carry_fields': [], 'sequence_fields': []}}},
    )

    assert result == {'report_name': 'Example', 'rows': [{'value': 'A'}]}
    assert trace[0] == {
        'action': 'SKIP_EMPTY_CONTINUATION',
        'file': '00006.json',
        'record_index': 0,
        'unit_name': None,
        'source': {'page_index': 5, 'table_index': 0},
        'row_count': 0,
        'carry_fields': [],
        'sequence_fields': [],
    }
    assert [entry['action'] for entry in trace[1:]] == ['NEW_PARENT', 'SUMMARY']


@pytest.mark.parametrize('rows', [[{'value': 'A'}], [{}], None, {}])
def test_page_aggregation_rejects_nonempty_or_malformed_anonymous_orphan(
    rows: object,
) -> None:
    with pytest.raises(ValueError, match='Orphan continuation'):
        aggregate_page_results(
            [
                (
                    '00006.json',
                    {
                        'records': [
                            {
                                'unit_name': None,
                                'continuation': {'is_continuation': True},
                                'rows': rows,
                            }
                        ]
                    },
                )
            ],
            {'units': {'rows': {'carry_fields': [], 'sequence_fields': []}}},
        )


def test_empty_named_record_establishes_owner_for_continuation() -> None:
    result, trace = aggregate_page_results(
        [
            (
                '00001.json',
                {
                    'records': [
                        {
                            'unit_name': 'rows',
                            'continuation': {'is_continuation': False},
                            'rows': [],
                        },
                        {
                            'unit_name': None,
                            'continuation': {'is_continuation': True},
                            'rows': [{'value': 'A'}],
                        },
                    ]
                },
            )
        ],
        {'units': {'rows': {'carry_fields': [], 'sequence_fields': []}}},
    )

    assert result == {'rows': [{'value': 'A'}]}
    assert [entry['action'] for entry in trace] == ['NEW_PARENT', 'MERGE', 'SUMMARY']


def test_empty_attempted_array_requests_targeted_repair() -> None:
    units = build_extraction_units(_schema())
    findings = verify_result(
        {'claim_number': 'ABC', 'payer': None, 'service_lines': []},
        _schema(),
        units,
    )
    assert [
        (finding.code, finding.unit_name, finding.repairable) for finding in findings
    ] == [('empty-array', 'service_lines', True)]


def test_deterministic_modules_do_not_import_model_runtime() -> None:
    ops_root = Path(__file__).resolve().parents[1] / 'src' / 'marie_longextract' / 'ops'
    forbidden = ('marie.agent', 'marie.engine', 'openai', 'litellm')
    for name in ('schema.py', 'stitch.py', 'verify.py'):
        source = (ops_root / name).read_text(encoding='utf-8')
        assert not any(module in source for module in forbidden)
