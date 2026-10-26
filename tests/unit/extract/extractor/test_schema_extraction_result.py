from __future__ import annotations

import copy

import pytest


def build(schema: dict, pages: list[dict], ocr: object = None) -> dict:
    from marie.extract.extractor.schema_extraction_result import (
        build_schema_extraction_result,
    )

    return build_schema_extraction_result(
        schema, pages, ocr, 'dag-1', 'configured-model'
    )


def test_merges_pages_in_source_order_and_preserves_repeated_scalar_identity() -> None:
    schema = {'type': 'object', 'properties': {'payer': {'type': 'string'}}}
    pages = [
        {'page': 7, 'data': {'payer': 'ACME'}},
        {'page': 2, 'data': {'payer': 'ACME'}},
    ]
    ocr = [
        {
            'meta': {'page': p, 'imageSize': {'width': 100, 'height': 200}},
            'words': [{'text': 'ACME', 'confidence': 0.99, 'box': [10, 20, 40, 10]}],
        }
        for p in [2, 7]
    ]
    result = build(schema, pages, ocr)
    assert result['dag_id'] == 'dag-1'
    assert result['model'] == 'configured-model'
    assert result['pages'] == {
        'extracted': 2,
        'total': 2,
        'sizes': [
            {'page': 2, 'width': 100, 'height': 200},
            {'page': 7, 'width': 100, 'height': 200},
        ],
    }
    first, second = result['values']
    assert first['field_id'] == second['field_id'] == 'payer'
    assert first['path'] == second['path'] == 'payer'
    assert [first['occurrence_index'], second['occurrence_index']] == [0, 1]
    assert first['id'] != second['id']
    assert first['row_index'] is None
    assert [first['location']['page'], second['location']['page']] == [2, 7]
    assert first['confidence'] is second['confidence'] is None
    assert result['missing_required'] == []


def test_nested_arrays_keep_paths_and_global_row_and_occurrence_indexes() -> None:
    schema = {
        'type': 'object',
        'properties': {
            'claims': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'claim_id': {'type': 'string'},
                        'lines': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {'amount': {'type': 'number'}},
                            },
                        },
                    },
                },
            }
        },
    }
    pages = [
        {
            'page': 1,
            'data': {'claims': [{'claim_id': 'B', 'lines': [{'amount': -50}]}]},
        },
        {
            'page': 0,
            'data': {
                'claims': [{'claim_id': 'A', 'lines': [{'amount': 1}, {'amount': 2}]}]
            },
        },
    ]
    original = copy.deepcopy(pages)
    result = build(schema, pages)
    assert [
        (v['path'], v['row_index'], v['occurrence_index'], v['value'])
        for v in result['values']
    ] == [
        ('claims[0].claim_id', 0, 0, 'A'),
        ('claims[0].lines[0].amount', 0, 0, '1'),
        ('claims[0].lines[1].amount', 1, 1, '2'),
        ('claims[1].claim_id', 1, 1, 'B'),
        ('claims[1].lines[0].amount', 2, 2, '-50'),
    ]
    assert result['values'][-1]['field_id'] == 'claims.lines.amount'
    assert all(v['location'] is None for v in result['values'])
    assert pages == original


def test_required_fields_are_evaluated_after_merging_page_objects() -> None:
    schema = {
        'type': 'object',
        'required': ['payer', 'patient', 'absent'],
        'properties': {
            'payer': {'type': 'string'},
            'absent': {'type': 'string'},
            'patient': {
                'type': 'object',
                'required': ['first', 'last'],
                'properties': {
                    'first': {'type': 'string'},
                    'last': {'type': 'string'},
                },
            },
        },
    }
    result = build(
        schema,
        [
            {'page': 0, 'data': {'patient': {'first': 'Alice'}}},
            {'page': 1, 'data': {'payer': 'ACME', 'patient': {'last': 'Jones'}}},
        ],
    )
    assert result['missing_required'] == ['absent']


def test_incomplete_required_rows_report_field_path_once() -> None:
    schema = {
        'type': 'object',
        'properties': {
            'lines': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'required': ['amount'],
                    'properties': {'amount': {'type': 'number'}},
                },
            }
        },
    }
    result = build(schema, [{'page': 0, 'data': {'lines': [{}, {}, {'amount': 0}]}}])
    assert result['missing_required'] == ['lines.amount']
    assert result['values'][0]['value'] == '0'
    assert result['values'][0]['row_index'] == 2


@pytest.mark.parametrize(
    'data',
    [
        {'amount': '714.57'},
        {'amount': -1},
        {'amount': 10, 'unknown': 1},
        {'amount': True},
    ],
)
def test_rejects_nonmissing_type_constraint_and_additional_property_errors(
    data: dict,
) -> None:
    schema = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {'amount': {'type': 'number', 'minimum': 0}},
    }
    with pytest.raises(ValueError, match='page 4'):
        build(schema, [{'page': 4, 'data': data}])


def test_local_refs_composition_enums_and_formats_are_validated() -> None:
    schema = {
        'type': 'object',
        '$defs': {'code': {'type': 'string', 'enum': ['A', 'B']}},
        'allOf': [{'required': ['code']}],
        'properties': {
            'code': {'$ref': '#/$defs/code'},
            'date': {'type': 'string', 'format': 'date'},
        },
    }
    assert (
        build(schema, [{'page': 0, 'data': {'code': 'A', 'date': '2026-10-26'}}])[
            'missing_required'
        ]
        == []
    )
    assert build(schema, [{'page': 0, 'data': {}}])['missing_required'] == ['code']
    with pytest.raises(ValueError, match='code'):
        build(schema, [{'page': 0, 'data': {'code': 'C'}}])
    with pytest.raises(ValueError, match='date'):
        build(schema, [{'page': 0, 'data': {'date': '2026-99-99'}}])


def test_optional_nullable_fields_and_primitive_arrays_need_no_extension_metadata() -> (
    None
):
    schema = {
        'type': 'object',
        'properties': {
            'note': {'type': ['string', 'null']},
            'tags': {'type': 'array', 'items': {'type': 'string'}},
            'approved': {'type': 'boolean'},
        },
    }
    result = build(
        schema,
        [{'page': 0, 'data': {'note': None, 'tags': ['a', 'b'], 'approved': False}}],
    )
    assert [(v['field_id'], v['path'], v['value']) for v in result['values']] == [
        ('tags', 'tags[0]', 'a'),
        ('tags', 'tags[1]', 'b'),
        ('approved', 'approved', 'false'),
    ]


def test_ordinary_oneof_required_branches_validate_without_disabling_required_keyword() -> (
    None
):
    schema = {
        'type': 'object',
        'properties': {'a': {'type': 'string'}, 'b': {'type': 'number'}},
        'oneOf': [{'required': ['a']}, {'required': ['b']}],
    }
    assert build(schema, [{'page': 0, 'data': {'a': 'x'}}])['missing_required'] == []
    with pytest.raises(ValueError, match='page 0'):
        build(schema, [{'page': 0, 'data': {'a': 'x', 'b': 1}}])


def test_rejects_invalid_schema_and_nonobject_or_nonfinite_page_output() -> None:
    with pytest.raises(ValueError, match='schema'):
        build({'type': 'invalid'}, [])
    with pytest.raises(ValueError, match='page 0'):
        build({'type': 'object'}, [{'page': 0, 'data': ['bad']}])
    with pytest.raises(ValueError, match='page 0'):
        build({'type': 'object'}, [{'page': 0, 'data': {'amount': float('nan')}}])


def test_page_counts_include_ocr_pages_without_extractions() -> None:
    ocr = [
        {'meta': {'page': p, 'imageSize': {'width': 100, 'height': 200}}, 'words': []}
        for p in range(3)
    ]
    result = build({'type': 'object'}, [{'page': 2, 'data': {}}], {'ocr': ocr})
    assert result['pages']['extracted'] == 1
    assert result['pages']['total'] == 3


def test_arrays_of_arrays_keep_row_indexes_separate_from_container_indexes() -> None:
    schema = {
        'type': 'object',
        'properties': {
            'matrix': {
                'type': 'array',
                'items': {'type': 'array', 'items': {'type': 'integer'}},
            }
        },
    }
    result = build(
        schema,
        [
            {'page': 0, 'data': {'matrix': [[1, 2]]}},
            {'page': 1, 'data': {'matrix': [[3]]}},
        ],
    )
    assert [(v['path'], v['row_index']) for v in result['values']] == [
        ('matrix[0][0]', 0),
        ('matrix[0][1]', 1),
        ('matrix[1][0]', 2),
    ]


@pytest.mark.parametrize(
    ('schema_type', 'value', 'printed', 'should_match'),
    [
        ('number', 17, '$17.00', True),
        ('number', 17.0, '$17.00', True),
        ('number', 1234, '$1,234.00', True),
        ('number', -50, '(50.00)', True),
        ('number', 714.57, '71457', False),
        ('number', -50, '50.00', False),
        ('string', '0017', '17', False),
        ('string', '17', '$17.00', False),
        ('string', '1/23', '12/3', False),
        ('boolean', True, '1.00', False),
    ],
)
def test_numeric_values_ground_equivalent_decimal_amounts_without_relaxing_identifiers(
    schema_type: str, value: object, printed: str, should_match: bool
) -> None:
    schema = {'type': 'object', 'properties': {'value': {'type': schema_type}}}
    ocr = [
        {
            'meta': {'page': 0, 'imageSize': {'width': 100, 'height': 100}},
            'words': [{'text': printed, 'box': [10, 20, 40, 10]}],
        }
    ]
    extracted = build(schema, [{'page': 0, 'data': {'value': value}}], ocr)['values'][0]
    if should_match:
        assert extracted['location'] == {
            'page': 0,
            'bbox': {'x1': 0.1, 'y1': 0.2, 'x2': 0.5, 'y2': 0.3},
            'match': 'exact',
            'candidates': 1,
        }
    else:
        assert extracted['location'] is None
    assert extracted['confidence'] is None
