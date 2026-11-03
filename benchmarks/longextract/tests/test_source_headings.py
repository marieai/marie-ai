from __future__ import annotations

import copy
import json

import pytest
from marie_longextract.ops.source_headings import (
    build_source_heading_prompt,
    heading_units,
    validate_source_headings,
)


@pytest.fixture
def units() -> list[dict]:
    return [
        {
            'unit_name': 'detail_rows',
            'description': 'Detail rows within each reporting area; exclude totals.',
            'fields': {
                'area': {
                    'type': 'string',
                    'description': 'The geography heading governing this detail row.',
                },
                'section': {
                    'type': ['string', 'null'],
                    'description': 'The preceding section-only heading with no values.',
                },
            },
        }
    ]


def test_heading_units_selects_descriptions_without_inferring_from_names() -> None:
    area = {'type': 'string', 'description': 'The governing geography HEADING.'}
    section = {
        'type': ['string', 'null'],
        'description': 'Use the printed section headings.',
    }
    schema_units = [
        {
            'unit_name': 'document_fields',
            'unit_kind': 'object',
            'schema': {'type': 'object', 'properties': {'area': area}},
        },
        {
            'unit_name': 'records',
            'unit_kind': 'array',
            'schema': {
                'type': 'array',
                'description': 'Include detail rows; exclude summary rows.',
                'items': {
                    'type': 'object',
                    'properties': {
                        'area': area,
                        'section': section,
                        'heading_name': {'type': 'string', 'description': 'A label.'},
                        'amount': {'type': 'integer', 'description': 'Heading total.'},
                        'axis': {'type': 'string', 'description': 'The matrix axis.'},
                    },
                },
            },
        },
        {
            'unit_name': 'unrelated',
            'unit_kind': 'array',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {'label': {'type': 'string'}},
                },
            },
        },
    ]
    original = copy.deepcopy(schema_units)

    assert heading_units(schema_units) == [
        {
            'unit_name': 'records',
            'description': 'Include detail rows; exclude summary rows.',
            'fields': {'area': area, 'section': section},
        }
    ]
    assert schema_units == original


def test_prompt_preserves_numbered_evidence_and_schema_scope(units: list[dict]) -> None:
    prior_context = {
        'previous_page': {
            'page_number': 4,
            'lines': [
                '17 | | AREA B | 9,000 | 20 |',
                '18 | | Household total | 8,000 |',
            ],
            'truncated': False,
        },
        'prior_table_title_candidates': [
            {'page_number': 2, 'line_number': 1, 'text': 'Table B. Area details'}
        ],
    }
    original = copy.deepcopy(prior_context)

    prompt = build_source_heading_prompt(
        5, '| First detail | 100 |\n| AREA C | 10,000 |', prior_context, units
    )
    evidence = json.loads(prompt.split('Source evidence (JSON):\n', 1)[1])

    assert evidence == {
        'page_number': 5,
        'page_text': '| First detail | 100 |\n| AREA C | 10,000 |',
        'prior_context': original,
        'units': units,
    }
    assert prior_context == original


def test_heading_request_bounds_current_rows_but_keeps_late_printed_titles(
    units: list[dict],
) -> None:
    text = '\n'.join(
        ['Opening detail | 1'] * 40
        + ['UNNEEDED DETAIL | 999'] * 200
        + ['Table Z. New source block']
    )
    prompt = build_source_heading_prompt(5, text, {}, units)
    evidence = json.loads(prompt.split('Source evidence (JSON):\n', 1)[1])

    assert len(evidence['page_text'].splitlines()) == 40
    assert 'UNNEEDED DETAIL' not in prompt
    assert evidence['current_table_title_candidates'] == [
        {'line_number': 241, 'text': 'Table Z. New source block'}
    ]
    assert evidence['page_text_truncated'] is True


@pytest.fixture
def page_texts() -> dict[int, str]:
    return {
        3: '| OLD AREA | 7,000 |',
        4: 'Prior detail 75\n  | AREA   B | 9,000 | 20 |  \nHousehold total 8,000',
        5: '| Current detail | 100 |\n| AREA C | 10,000 |',
    }


@pytest.fixture
def heading() -> dict:
    return {
        'unit_name': 'detail_rows',
        'field_name': 'area',
        'value': 'AREA B',
        'source_page': 4,
        'line_number': 2,
        'source_quote': '| AREA   B | 9,000 | 20 |',
    }


def test_validation_accepts_a_numeric_heading_with_full_provenance(
    heading: dict, page_texts: dict[int, str], units: list[dict]
) -> None:
    payload = {'headings': [{**heading, 'value': '  AREA  B\t'}]}
    original = copy.deepcopy(payload)

    assert validate_source_headings(payload, 5, page_texts, units) == [heading]
    assert payload == original


def test_heading_quote_may_omit_outer_table_borders_only(
    heading: dict, page_texts: dict[int, str], units: list[dict]
) -> None:
    payload = {'headings': [{**heading, 'source_quote': 'AREA   B | 9,000 | 20'}]}

    assert validate_source_headings(payload, 5, page_texts, units) == [heading]


@pytest.mark.parametrize(
    ('line', 'value'),
    [
        ('AREA   B  9,000 20', 'AREA B'),
        ('| AREA B|9,000|20 |', 'AREA B'),
        ('Reporting area: North-East (Outer)', 'North-East (Outer)'),
    ],
)
def test_validation_accepts_whole_tokens_or_a_complete_pipe_cell(
    line: str, value: str, heading: dict, units: list[dict]
) -> None:
    item = {**heading, 'value': value, 'line_number': 1, 'source_quote': line}

    assert validate_source_headings({'headings': [item]}, 5, {4: line}, units) == [item]


@pytest.mark.parametrize(
    ('changes', 'error'),
    [
        ({'unit_name': 'other_rows'}, 'unit or field'),
        ({'field_name': 'language'}, 'unit or field'),
        ({'value': None}, 'non-empty string'),
        ({'value': 9000}, 'non-empty string'),
        ({'value': '   '}, 'non-empty string'),
        ({'value': 'REA B'}, 'whole source tokens'),
        ({'value': 'Area B'}, 'whole source tokens'),
        ({'value': 'AREA-B'}, 'whole source tokens'),
        ({'value': 'AREA C'}, 'whole source tokens'),
        ({'source_page': 3}, 'immediately preceding page'),
        ({'source_page': 5}, 'immediately preceding page'),
        ({'source_page': 6}, 'immediately preceding page'),
        ({'source_page': True}, 'immediately preceding page'),
        ({'source_page': 4.0}, 'immediately preceding page'),
        ({'line_number': 0}, 'source line'),
        ({'line_number': 4}, 'source line'),
        ({'line_number': True}, 'source line'),
        ({'line_number': 2.0}, 'source line'),
        ({'source_quote': 'AREA   B'}, 'complete printed source line'),
        ({'source_quote': '| AREA B | 9,000 | 20 |'}, 'complete printed source line'),
        ({'source_quote': None}, 'complete printed source line'),
    ],
)
def test_validation_rejects_ungrounded_model_fields(
    changes: dict,
    error: str,
    heading: dict,
    page_texts: dict[int, str],
    units: list[dict],
) -> None:
    with pytest.raises(ValueError, match=error):
        validate_source_headings(
            {'headings': [{**heading, **changes}]}, 5, page_texts, units
        )


def test_validation_rejects_duplicate_field_resolutions(
    heading: dict, page_texts: dict[int, str], units: list[dict]
) -> None:
    with pytest.raises(ValueError, match='Duplicate'):
        validate_source_headings({'headings': [heading, heading]}, 5, page_texts, units)


def test_validation_rejects_missing_source_page(
    heading: dict, units: list[dict]
) -> None:
    with pytest.raises(ValueError, match='source page'):
        validate_source_headings({'headings': [heading]}, 5, {}, units)


@pytest.mark.parametrize(
    'payload', [None, [], {}, {'headings': None}, {'headings': [None]}]
)
def test_validation_rejects_malformed_responses(
    payload: object, units: list[dict]
) -> None:
    with pytest.raises(ValueError):
        validate_source_headings(payload, 5, {4: ''}, units)


def test_validation_allows_an_empty_result_on_the_first_page(units: list[dict]) -> None:
    assert validate_source_headings({'headings': []}, 1, {}, units) == []
