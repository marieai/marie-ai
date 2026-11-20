from __future__ import annotations

import copy

import pytest
from marie_longextract.ops.refinement import select_refined_pages


@pytest.fixture
def units() -> list[dict]:
    return [
        {
            'unit_name': 'document_fields',
            'unit_kind': 'object',
            'schema': {
                'type': 'object',
                'properties': {'release': {'type': 'string'}},
                'required': ['release'],
            },
        },
        {
            'unit_name': 'details',
            'unit_kind': 'array',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'heading': {'type': 'string'},
                        'amount': {'type': 'integer'},
                    },
                    'required': ['heading', 'amount'],
                    'additionalProperties': False,
                },
            },
        },
        {
            'unit_name': 'notes',
            'unit_kind': 'array',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {'text': {'type': 'string'}},
                    'required': ['text'],
                    'additionalProperties': False,
                },
            },
        },
    ]


def page(rows: list, owner: str | None = 'details', continuation: bool = False) -> dict:
    return {
        'document_fields': {},
        'records': [
            {
                'unit_name': owner,
                'continuation': {'is_continuation': continuation},
                'rows': rows,
            }
        ],
    }


def test_empty_initial_page_rejects_new_rows_with_null_required_headings(
    units: list[dict],
) -> None:
    initial = [('00001.json', {'records': []})]
    candidate = [('00001.json', page([{'heading': None, 'amount': 17}]))]

    selected, decisions = select_refined_pages(initial, candidate, units)

    assert selected == initial
    assert len(decisions) == 1
    assert decisions[0]['filename'] == '00001.json'
    assert decisions[0]['selected'] == 'initial'
    assert decisions[0]['initial_error_count'] == 0
    assert decisions[0]['candidate_error_count'] == 1
    assert decisions[0]['candidate_error_examples'][0]['path'] == [
        'details',
        0,
        'heading',
    ]


def test_empty_initial_page_retains_new_schema_valid_rows(units: list[dict]) -> None:
    initial = [('00001.json', {'records': []})]
    candidate = [('00001.json', page([{'heading': 'GROUP A', 'amount': 17}]))]

    selected, decisions = select_refined_pages(initial, candidate, units)

    assert selected == candidate
    assert decisions == []


def test_schema_valid_value_corrections_and_added_rows_are_preserved(
    units: list[dict],
) -> None:
    initial = [('00001.json', page([{'heading': 'GROUP A', 'amount': 10}]))]
    candidate = [
        (
            '00001.json',
            page(
                [
                    {'heading': 'GROUP A', 'amount': 11},
                    {'heading': 'GROUP B', 'amount': 15},
                ]
            ),
        )
    ]

    assert select_refined_pages(initial, candidate, units) == (candidate, [])


def test_candidate_with_fewer_errors_is_selected(units: list[dict]) -> None:
    initial = [('00001.json', page([{'heading': None, 'amount': 'bad'}]))]
    candidate = [('00001.json', page([{'heading': 'GROUP A', 'amount': 'bad'}]))]

    assert select_refined_pages(initial, candidate, units) == (candidate, [])


def test_continuation_uses_the_preceding_selected_owner(units: list[dict]) -> None:
    initial = [
        ('00002.json', page([{'heading': 'GROUP A', 'amount': 9}], None, True)),
        ('00001.json', page([{'heading': 'GROUP A', 'amount': 8}])),
    ]
    candidate = [
        ('00001.json', page([{'text': None}], 'notes')),
        ('00002.json', page([{'heading': 'GROUP A', 'amount': 10}], None, True)),
    ]

    selected, decisions = select_refined_pages(initial, candidate, units)

    assert selected == [initial[1], candidate[1]]
    assert [decision['filename'] for decision in decisions] == ['00001.json']


def test_accepted_new_owner_governs_subsequent_unnamed_continuations(
    units: list[dict],
) -> None:
    initial = [
        ('00001.json', page([{'heading': 'GROUP A', 'amount': 8}])),
        ('00002.json', page([{'heading': 'GROUP A', 'amount': 9}], None, True)),
    ]
    candidate = [
        ('00001.json', page([{'text': 'A note'}], 'notes')),
        ('00002.json', page([{'text': 'Continued note'}], None, True)),
    ]

    assert select_refined_pages(initial, candidate, units) == (candidate, [])


def test_explicit_continuation_owner_is_used_without_row_shape_inference(
    units: list[dict],
) -> None:
    initial = [('00001.json', {'records': []})]
    candidate = [('00001.json', page([{'text': 'A note'}], 'notes', True))]

    assert select_refined_pages(initial, candidate, units) == (candidate, [])


@pytest.mark.parametrize(
    'invalid_page',
    [
        page([{'heading': 'GROUP A', 'amount': 9}], 'unknown'),
        page([{'heading': 'GROUP A', 'amount': 9}], None, True),
        page([{'heading': 'GROUP A', 'amount': 9}], None, False),
        page(['not an object']),
        page([{'heading': 'GROUP A', 'amount': '9'}]),
        page([{'heading': 'GROUP A'}]),
        page([{'heading': 'GROUP A', 'amount': 9, 'invented': True}]),
        {'records': None},
        {'records': [None]},
        {
            'records': [
                {
                    'unit_name': 'details',
                    'continuation': {'is_continuation': False},
                    'rows': None,
                }
            ]
        },
        {
            'records': [
                {
                    'unit_name': 'details',
                    'continuation': {'is_continuation': 'false'},
                    'rows': [],
                }
            ]
        },
    ],
)
def test_candidate_structural_or_row_schema_errors_cause_fallback(
    invalid_page: dict, units: list[dict]
) -> None:
    initial = [('00001.json', {'records': []})]

    selected, decisions = select_refined_pages(
        initial, [('00001.json', invalid_page)], units
    )

    assert selected == initial
    assert decisions[0]['candidate_error_count'] > 0
    assert decisions[0]['candidate_error_examples']


def test_unnamed_continuation_is_not_reclassified_by_its_rows(
    units: list[dict],
) -> None:
    initial = [
        ('00001.json', page([{'heading': 'GROUP A', 'amount': 8}])),
        ('00002.json', {'records': []}),
    ]
    candidate = [initial[0], ('00002.json', page([{'text': 'A note'}], None, True))]

    selected, decisions = select_refined_pages(initial, candidate, units)

    assert selected == initial
    assert decisions[0]['filename'] == '00002.json'


def test_ownership_can_change_between_records_within_a_page(units: list[dict]) -> None:
    candidate = page([{'heading': 'GROUP A', 'amount': 8}])
    candidate['records'].extend(page([{'text': 'A note'}], 'notes')['records'])
    candidate['records'].extend(page([{'text': 'More notes'}], None, True)['records'])

    assert select_refined_pages(
        [('00001.json', {'records': []})], [('00001.json', candidate)], units
    ) == ([('00001.json', candidate)], [])


def test_pass_inputs_are_not_mutated(units: list[dict]) -> None:
    initial = [('00001.json', page([{'heading': 'GROUP A', 'amount': 8}]))]
    candidate = [('00001.json', page([{'heading': None, 'amount': 8}]))]
    before = copy.deepcopy((initial, candidate, units))

    select_refined_pages(initial, candidate, units)

    assert (initial, candidate, units) == before


@pytest.mark.parametrize(
    ('initial', 'candidate'),
    [
        ([('00001.json', {'records': []})], []),
        ([], [('00001.json', {'records': []})]),
        ([('00001.json', {'records': []})], [('00002.json', {'records': []})]),
        ([('00001.json', {'records': []})] * 2, [('00001.json', {'records': []})]),
        ([('00001.json', {'records': []})], [('00001.json', {'records': []})] * 2),
    ],
)
def test_mismatched_or_duplicate_filenames_fail(
    initial: list, candidate: list, units: list[dict]
) -> None:
    with pytest.raises(ValueError, match='filenames'):
        select_refined_pages(initial, candidate, units)


def test_empty_passes_are_allowed(units: list[dict]) -> None:
    assert select_refined_pages([], [], units) == ([], [])


def test_explicit_continuation_owner_cannot_contradict_the_active_unit(
    units: list[dict],
) -> None:
    initial = [
        ('00001.json', page([{'heading': 'GROUP A', 'amount': 8}])),
        ('00002.json', {'records': []}),
    ]
    candidate = [initial[0], ('00002.json', page([{'text': 'A note'}], 'notes', True))]

    selected, decisions = select_refined_pages(initial, candidate, units)

    assert selected == initial
    assert decisions[0]['filename'] == '00002.json'
    assert decisions[0]['candidate_error_count'] == 1
    assert (
        'contradicts active unit'
        in decisions[0]['candidate_error_examples'][0]['message']
    )


def test_valid_new_rows_are_checked_after_heading_carry_without_mutating_raw_passes(
    units: list[dict],
) -> None:
    initial = [
        ('00001.json', page([{'heading': 'GROUP A', 'amount': 8}])),
        ('00002.json', {'records': []}),
    ]
    candidate = [
        initial[0],
        ('00002.json', page([{'heading': None, 'amount': 9}], None, True)),
    ]
    policy = {
        'units': {'details': {'carry_fields': ['heading'], 'sequence_fields': []}}
    }
    original = copy.deepcopy((initial, candidate, units, policy))

    assert select_refined_pages(initial, candidate, units, policy) == (candidate, [])
    assert (initial, candidate, units, policy) == original
    assert candidate[1][1]['records'][0]['rows'][0]['heading'] is None


def test_sequence_fields_are_validated_after_document_wide_assignment(
    units: list[dict],
) -> None:
    item_schema = units[1]['schema']['items']
    item_schema['properties']['ordinal'] = {'type': 'integer', 'minimum': 1}
    item_schema['required'].append('ordinal')
    initial = [
        ('00001.json', page([{'heading': 'GROUP A', 'amount': 8, 'ordinal': 0}])),
        ('00002.json', {'records': []}),
    ]
    candidate = [
        initial[0],
        (
            '00002.json',
            page([{'heading': 'GROUP A', 'amount': 9, 'ordinal': 0}], None, True),
        ),
    ]
    policy = {
        'units': {'details': {'carry_fields': [], 'sequence_fields': ['ordinal']}}
    }

    assert select_refined_pages(initial, candidate, units, policy) == (candidate, [])
    assert candidate[1][1]['records'][0]['rows'][0]['ordinal'] == 0


def test_aggregation_failure_is_worse_than_multiple_row_schema_errors(
    units: list[dict],
) -> None:
    initial = [
        ('00001.json', page([{'heading': 'GROUP A', 'amount': 8}])),
        ('00002.json', page([{'heading': None, 'amount': 'bad'}], None, True)),
    ]
    candidate = [initial[0], ('00002.json', page([{'text': 'A note'}], 'notes', True))]

    selected, decisions = select_refined_pages(initial, candidate, units)

    assert selected == initial
    assert decisions[0]['initial_error_count'] == 2
    assert decisions[0]['candidate_error_count'] == 1


def test_empty_orphan_continuation_uses_the_aggregators_skip_rule(
    units: list[dict],
) -> None:
    initial = [('00001.json', {'records': []})]
    candidate = [('00001.json', page([], None, True))]

    assert select_refined_pages(initial, candidate, units) == (candidate, [])
