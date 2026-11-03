from __future__ import annotations

import copy

import pytest
from marie_longextract.ops.page_context import audit_table_continuations


@pytest.mark.parametrize(
    ('previous', 'current', 'detected', 'expected'),
    [
        ('Table A. Totals', 'Table B. Detail--Continues', 'Table B. Detail', False),
        ('Table B. Detail--Continues', 'Table B. Detail--Table Ends', 'Table B', True),
        ('Table A. Totals', 'continued detail', 'Table B', True),
        ('Table A. Totals', 'Table B. Detail', 'Table 1', True),
        ('Table A. Totals', 'Table B. Detail\nTable C. Notes', 'Table B', True),
        ('Table A. Totals', 'Table B. Detail\n2 of 3', 'Table B', True),
    ],
)
def test_printed_identity_audit_changes_only_an_unambiguous_new_table(
    previous: str, current: str, detected: str, expected: bool
) -> None:
    tables = [
        {
            'name': detected,
            'continuation': {
                'is_continuation': True,
                'continuation_rationale': 'Title says continues',
            },
            'rows': [{'line_number': 3}],
        }
    ]
    original = copy.deepcopy(tables)
    context = {
        'previous_page': {'page_number': 1},
        'prior_table_title_candidates': [
            {'text': previous, 'page_number': 1, 'line_number': 1}
        ],
    }

    audited = audit_table_continuations(tables, current, context)

    assert audited[0]['continuation']['is_continuation'] is expected
    assert audited[0]['rows'] == original[0]['rows']
    assert tables == original
    if expected is False:
        assert previous in audited[0]['continuation']['continuation_rationale']
        assert current in audited[0]['continuation']['continuation_rationale']


def test_printed_identity_audit_does_not_guess_between_multiple_detected_tables() -> (
    None
):
    tables = [
        {'name': 'Table B', 'continuation': {'is_continuation': True}},
        {'name': 'Table C', 'continuation': {'is_continuation': True}},
    ]
    context = {
        'previous_page': {'page_number': 1},
        'prior_table_title_candidates': [
            {'text': 'Table A. Totals', 'page_number': 1, 'line_number': 1}
        ],
    }

    assert audit_table_continuations(tables, 'Table B. Detail', context) == tables


def test_title_reference_at_previous_page_end_cannot_reset_a_continuation() -> None:
    from marie_longextract.ops.page_context import build_previous_page_context

    context = build_previous_page_context(
        2,
        {
            1: 'Table A. Regional totals\nData | 100\nTable B summarizes these results in the appendix.'
        },
    )
    tables = [
        {
            'name': 'Table A. Regional totals--Table Ends',
            'continuation': {'is_continuation': True},
        }
    ]

    assert (
        audit_table_continuations(
            tables, 'Table A. Regional totals--Table Ends', context
        )
        == tables
    )
