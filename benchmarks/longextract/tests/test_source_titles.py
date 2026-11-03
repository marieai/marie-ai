from __future__ import annotations

import copy

import pytest


def _units() -> list[dict]:
    return [
        {
            'unit_name': 'measurements',
            'unit_kind': 'array',
            'schema': {
                'description': 'One row for each measurement.',
                'items': {
                    'properties': {
                        'cohort': {
                            'type': 'string',
                            'description': 'Population printed in the table title.',
                        },
                        'site': {
                            'type': 'string',
                            'description': 'Location printed in the table title.',
                        },
                        'sample_count': {
                            'type': 'integer',
                            'description': 'Count printed in the table title.',
                        },
                        'value': {
                            'type': 'number',
                            'description': 'Value in the detail row.',
                        },
                    }
                },
            },
        }
    ]


def _payload() -> dict:
    return {
        'titles': [
            {
                'unit_name': 'measurements',
                'source_page': 2,
                'line_start': 1,
                'line_end': 2,
                'source_quote': 'Measurements - Adults 18+\nWrist (N=12)',
                'fields': {'cohort': 'Adults 18+', 'site': 'Wrist', 'sample_count': 12},
            }
        ]
    }


def test_select_title_fields_without_detail_fields() -> None:
    from marie_longextract.ops.source_titles import title_units

    selected = title_units(_units())
    assert set(selected[0]['fields']) == {'cohort', 'site', 'sample_count'}
    assert selected[0]['description'] == 'One row for each measurement.'


def test_title_attributes_have_current_page_source_citations() -> None:
    from marie_longextract.ops.source_titles import title_units, validate_source_titles

    payload = _payload()
    before = copy.deepcopy(payload)
    actual = validate_source_titles(
        payload,
        2,
        'Measurements - Adults 18+\nWrist (N=12)\nA | 3',
        title_units(_units()),
    )
    assert actual == payload['titles']
    assert payload == before


def test_missing_required_title_attribute_is_rejected() -> None:
    from marie_longextract.ops.source_titles import title_units, validate_source_titles

    units = _units()
    units[0]['schema']['items']['required'] = ['site', 'value']
    payload = _payload()
    del payload['titles'][0]['fields']['site']
    with pytest.raises(ValueError, match='Missing required title'):
        validate_source_titles(
            payload, 2, 'Measurements - Adults 18+\nWrist (N=12)', title_units(units)
        )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('site', 'Elbow'),
        ('cohort', 'Adult'),
        ('sample_count', 13),
        ('sample_count', '12'),
        ('value', 3),
    ],
)
def test_reject_unsupported_or_unrequested_title_values(
    field: str, value: object
) -> None:
    from marie_longextract.ops.source_titles import title_units, validate_source_titles

    payload = _payload()
    payload['titles'][0]['fields'][field] = value
    with pytest.raises(ValueError):
        validate_source_titles(
            payload, 2, 'Measurements - Adults 18+\nWrist (N=12)', title_units(_units())
        )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('source_page', 3),
        ('line_start', 0),
        ('line_end', 20),
        ('source_quote', 'Adults 18+ Wrist'),
    ],
)
def test_reject_invalid_title_source(field: str, value: object) -> None:
    from marie_longextract.ops.source_titles import title_units, validate_source_titles

    payload = _payload()
    payload['titles'][0][field] = value
    with pytest.raises(ValueError):
        validate_source_titles(
            payload, 2, 'Measurements - Adults 18+\nWrist (N=12)', title_units(_units())
        )
