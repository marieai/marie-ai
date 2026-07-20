from __future__ import annotations

import pytest
from marie_longextract.agents.repair import (
    PageLeafRepairDecision,
    leaf_patches_from_decision,
    validate_leaf_repair_decision,
)


def _validate(decision: PageLeafRepairDecision, page_text: str, value: str) -> None:
    validate_leaf_repair_decision(
        decision,
        page_result=_page_result(value),
        schema={
            'properties': {
                'rows': {'items': {'properties': {'value': {'type': 'string'}}}}
            }
        },
        resolved_units=['rows'],
        current_page_text=page_text,
        previous_page_text='',
        allowed_targets={(0, 0, 'value', value)},
    )


def _page_result(value: str) -> dict[str, object]:
    return {
        'records': [
            {
                'continuation': {'is_continuation': False},
                'rows': [{'value': value}],
            }
        ]
    }


def _decision(value: str, **evidence: object) -> PageLeafRepairDecision:
    return PageLeafRepairDecision.model_validate(
        {
            'page_file': '00001.json',
            'reviews': [
                {
                    'record_index': 0,
                    'row_index': 0,
                    'field_name': 'value',
                    'action': 'use_source_candidate',
                    'evidence_source': 'current_page_text',
                    'rationale': 'The source span supplies the value.',
                    **evidence,
                }
            ],
        }
    )


def test_token_evidence_preserves_punctuation_without_symbol_rules() -> None:
    decision = _decision(value='X', evidence_line=1, evidence_quote='X')

    with pytest.raises(ValueError, match='line candidate'):
        _validate(decision, 'Total (X)', 'X')


def test_ordered_multiline_source_span_is_supported() -> None:
    value = 'Wrapped value continues here'
    decision = _decision(
        value=value,
        evidence_line=1,
        evidence_quote='Wrapped value',
        additional_evidence=[{'line': 2, 'quote': 'continues here'}],
        join_with=' ',
    )

    _validate(decision, 'Wrapped value\ncontinues here', value)

    assert (
        leaf_patches_from_decision(
            decision,
            page_result=_page_result(value),
        )
        == []
    )


def test_multiline_source_span_must_be_ordered() -> None:
    value = 'continues here Wrapped value'
    decision = _decision(
        value=value,
        evidence_line=2,
        evidence_quote='continues here',
        additional_evidence=[{'line': 1, 'quote': 'Wrapped value'}],
        join_with=' ',
    )

    with pytest.raises(ValueError, match='unique and ordered'):
        _validate(decision, 'Wrapped value\ncontinues here', value)
