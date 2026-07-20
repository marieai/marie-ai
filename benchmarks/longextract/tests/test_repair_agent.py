from __future__ import annotations

import json

import pytest
from marie_longextract.agents.repair import (
    PageLeafRepairDecision,
    RepairDecision,
    build_leaf_repair_prompt,
    build_repair_prompt,
    leaf_patches_from_decision,
    record_patch_from_decision,
    select_leaf_repair_consensus,
    validate_decision_evidence,
    validate_leaf_repair_decision,
)
from marie_longextract.ops.aggregation import aggregate_page_results
from marie_longextract.ops.repair import (
    apply_record_patch,
    apply_row_leaf_patches,
    infer_section_heading_patches,
)
from pydantic import ValidationError


def test_repair_decision_enforces_typed_continuation_boundary() -> None:
    with pytest.raises(ValidationError):
        RepairDecision.model_validate(
            {
                'page_file': '00029.json',
                'record_index': 0,
                'boundary': {
                    'kind': 'same_schema_unit',
                    'is_continuation': False,
                    'unit_name': 'housing_rows',
                    'new_unit_marker': None,
                },
                'sequence_evidence': ['page 2 of 2'],
                'schema_evidence': ['same enclosing profile'],
                'evidence': ['current page image'],
                'rationale': 'The table continues the preceding page.',
            }
        )


def test_record_patch_repairs_boundary_before_existing_aggregation() -> None:
    policy = {
        'units': {
            'demographic_rows': {'carry_fields': [], 'sequence_fields': []},
            'housing_rows': {'carry_fields': [], 'sequence_fields': []},
        }
    }
    first_page = {
        'document_fields': {},
        'records': [
            {
                'unit_name': 'demographic_rows',
                'source': {'page_index': 27, 'table_index': 0},
                'continuation': {'is_continuation': False},
                'rows': [{'label': 'Occupied housing units'}],
            }
        ],
    }
    second_page = {
        'document_fields': {},
        'records': [
            {
                'unit_name': 'housing_rows',
                'source': {'page_index': 28, 'table_index': 0},
                'continuation': {'is_continuation': False},
                'rows': [{'label': 'Total housing units'}],
            }
        ],
    }

    repaired_page = apply_record_patch(
        second_page,
        record_index=0,
        is_continuation=True,
        unit_name=None,
        active_unit='demographic_rows',
        allowed_units=policy['units'],
    )
    result, trace = aggregate_page_results(
        [('00028.json', first_page), ('00029.json', repaired_page)],
        policy,
    )

    assert result['demographic_rows'] == [
        {'label': 'Occupied housing units'},
        {'label': 'Total housing units'},
    ]
    assert 'housing_rows' not in result
    assert [entry['action'] for entry in trace] == [
        'NEW_PARENT',
        'MERGE',
        'SUMMARY',
    ]


def test_repair_decision_derives_patch_from_current_record() -> None:
    decision = RepairDecision.model_validate(
        {
            'page_file': '00029.json',
            'record_index': 0,
            'boundary': {
                'kind': 'same_schema_unit',
                'is_continuation': True,
                'unit_name': None,
                'new_unit_marker': None,
            },
            'sequence_evidence': ['page 3 of 3'],
            'schema_evidence': ['same enclosing profile'],
            'evidence': ['00027.png_prompt.txt'],
            'rationale': 'The logical profile continues.',
        }
    )

    patch = record_patch_from_decision(
        decision,
        {
            'unit_name': 'housing_rows',
            'continuation': {'is_continuation': False},
        },
    )

    assert patch is not None
    assert patch.is_continuation is True
    assert patch.unit_name is None


def test_repair_prompt_points_agent_at_stable_job_artifacts(tmp_path) -> None:
    paths = [
        'agent-output/longextract-unit-extract/00027.json',
        'agent-output/longextract-unit-extract/00027.png_prompt.txt',
        'agent-output/longextract-unit-extract/00028.json',
        'agent-output/longextract-unit-extract/00028.png_prompt.txt',
        'agent-output/longextract-unit-extract/00029.json',
        'agent-output/tables/00029.json',
    ]
    for relative_path in paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            'records': [
                {
                    'unit_name': 'demographic_rows',
                    'continuation': {'is_continuation': False},
                }
            ]
        }
        path.write_text(
            (
                json.dumps(value)
                if path.suffix == '.json'
                else '\nPage text:\nDP05 profile page 1 of 3\n\nPage tables:\n[]'
            ),
            encoding='utf-8',
        )

    prompt = build_repair_prompt(
        tmp_path,
        page_file='00029.json',
        record_index=0,
        active_unit='demographic_rows',
    )

    assert str(tmp_path) not in prompt
    assert 'agent-output/longextract-unit-extract/00028.json' in prompt
    assert 'agent-output/longextract-unit-extract/00029.json' in prompt
    assert 'agent-output/tables/00029.json' in prompt
    assert 'active-unit origin prompt' in prompt
    assert '00027.png_prompt.txt' in prompt
    assert 'This is not the same as a visual table boundary' in prompt
    assert 'table annotation is supporting evidence' in prompt
    assert 'descriptions of both the declared unit and the active unit' in prompt
    assert 'Existing unit_name and continuation values are hypotheses' in prompt
    assert 'Do not choose a unit from an individual row label' in prompt
    assert 'Active-unit origin prepared page text' in prompt


def test_new_unit_marker_must_be_grounded_in_current_page() -> None:
    decision = RepairDecision.model_validate(
        {
            'page_file': '00029.json',
            'record_index': 0,
            'boundary': {
                'kind': 'new_schema_unit',
                'is_continuation': False,
                'unit_name': 'housing_rows',
                'new_unit_marker': 'Invented profile title',
            },
            'sequence_evidence': ['current page'],
            'schema_evidence': ['new profile'],
            'evidence': ['00029.png_prompt.txt'],
            'rationale': 'A new profile starts here.',
        }
    )

    with pytest.raises(ValueError, match='not grounded'):
        validate_decision_evidence(
            decision,
            current_page_text='Total housing units',
            current_record={'rows': []},
        )


def test_new_unit_marker_cannot_be_a_record_value() -> None:
    decision = RepairDecision.model_validate(
        {
            'page_file': '00029.json',
            'record_index': 0,
            'boundary': {
                'kind': 'new_schema_unit',
                'is_continuation': False,
                'unit_name': 'housing_rows',
                'new_unit_marker': 'Total housing units',
            },
            'sequence_evidence': ['current page'],
            'schema_evidence': ['new profile'],
            'evidence': ['00029.png_prompt.txt'],
            'rationale': 'A new profile starts here.',
        }
    )

    with pytest.raises(ValueError, match='current record'):
        validate_decision_evidence(
            decision,
            current_page_text='Subject\nTotal housing units',
            current_record={'rows': [{'subject_label': 'Total housing units'}]},
        )


def test_string_leaf_repair_is_grounded_and_applied() -> None:
    page_result = {
        'records': [
            {
                'unit_name': 'rows',
                'continuation': {'is_continuation': False},
                'rows': [{'label': 'Total', 'display_value': 'X'}],
            }
        ]
    }
    schema = {
        'properties': {
            'rows': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'label': {'type': 'string'},
                        'display_value': {'type': ['string', 'null']},
                    },
                },
            }
        }
    }
    decision = PageLeafRepairDecision.model_validate(
        {
            'page_file': '00001.json',
            'reviews': [
                {
                    'record_index': 0,
                    'row_index': 0,
                    'field_name': 'display_value',
                    'action': 'use_source_candidate',
                    'evidence_source': 'current_page_text',
                    'evidence_line': 1,
                    'evidence_quote': '(X)',
                    'rationale': 'The printed delimiter is part of the value.',
                }
            ],
            'rationale': 'One unsupported string leaf was found.',
        }
    )

    validate_leaf_repair_decision(
        decision,
        page_result=page_result,
        schema=schema,
        resolved_units=['rows'],
        current_page_text='Total (X)',
        previous_page_text='',
        allowed_targets={(0, 0, 'display_value', 'X')},
    )
    patches = leaf_patches_from_decision(decision, page_result=page_result)
    repaired = apply_row_leaf_patches(
        page_result,
        [patch.model_dump(mode='python') for patch in patches],
    )

    assert repaired['records'][0]['rows'][0]['display_value'] == '(X)'
    assert page_result['records'][0]['rows'][0]['display_value'] == 'X'


def test_leaf_repair_consensus_selects_target_level_majorities() -> None:
    replacement = {
        'record_index': 0,
        'row_index': 0,
        'field_name': 'display_value',
        'action': 'use_source_candidate',
        'evidence_source': 'current_page_text',
        'evidence_line': 1,
        'evidence_quote': '(X)',
        'rationale': 'The delimiters are printed.',
    }
    minority = {
        'record_index': 0,
        'row_index': 1,
        'field_name': 'display_value',
        'action': 'use_source_candidate',
        'evidence_source': 'current_page_text',
        'evidence_line': 2,
        'evidence_quote': '(Y)',
        'rationale': 'The delimiters are printed.',
    }
    keep_x = {**replacement, 'evidence_quote': 'X'}
    keep_y = {**minority, 'evidence_quote': 'Y'}
    decisions = [
        PageLeafRepairDecision(page_file='00001.json', reviews=[replacement, minority]),
        PageLeafRepairDecision(page_file='00001.json', reviews=[replacement, keep_y]),
        PageLeafRepairDecision(page_file='00001.json', reviews=[keep_x, keep_y]),
    ]

    consensus = select_leaf_repair_consensus(
        page_file='00001.json',
        decisions=decisions,
        allowed_targets={
            (0, 0, 'display_value', 'X'),
            (0, 1, 'display_value', 'Y'),
        },
    )

    assert len(consensus.reviews) == 2
    page_result = {
        'records': [
            {
                'rows': [
                    {'display_value': 'X'},
                    {'display_value': 'Y'},
                ]
            }
        ]
    }
    assert [
        patch.model_dump()
        for patch in leaf_patches_from_decision(
            consensus,
            page_result=page_result,
        )
    ] == [
        {
            **{
                key: value
                for key, value in replacement.items()
                if key not in {'action', 'evidence_line'}
            },
            'expected_value': 'X',
            'replacement_value': '(X)',
        }
    ]


def test_leaf_repair_consensus_rejects_split_decisions() -> None:
    def decision(replacement_value: str | None) -> PageLeafRepairDecision:
        return PageLeafRepairDecision.model_validate(
            {
                'page_file': '00001.json',
                'reviews': [
                    {
                        'record_index': 0,
                        'row_index': 0,
                        'field_name': 'display_value',
                        'action': (
                            'clear_for_parser_carry'
                            if replacement_value is None
                            else 'use_source_candidate'
                        ),
                        'evidence_source': 'current_page_text',
                        'evidence_line': 1,
                        'evidence_quote': replacement_value or 'Total value',
                        'rationale': 'The source supports this value.',
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match='did not reach a majority'):
        select_leaf_repair_consensus(
            page_file='00001.json',
            decisions=[decision('(X)'), decision('[X]'), decision(None)],
            allowed_targets={(0, 0, 'display_value', 'X')},
        )


def test_string_leaf_repair_rejects_ungrounded_replacement() -> None:
    decision = PageLeafRepairDecision.model_validate(
        {
            'page_file': '00001.json',
            'reviews': [
                {
                    'record_index': 0,
                    'row_index': 0,
                    'field_name': 'heading',
                    'action': 'use_source_candidate',
                    'evidence_source': 'current_page_text',
                    'evidence_line': 1,
                    'evidence_quote': 'Invented heading',
                    'rationale': 'A heading should be corrected.',
                }
            ],
            'rationale': 'One patch was proposed.',
        }
    )

    with pytest.raises(ValueError, match='line candidate'):
        validate_leaf_repair_decision(
            decision,
            page_result={
                'records': [
                    {
                        'unit_name': 'rows',
                        'continuation': {'is_continuation': False},
                        'rows': [{'heading': 'Old heading'}],
                    }
                ]
            },
            schema={
                'properties': {
                    'rows': {
                        'items': {
                            'properties': {'heading': {'type': ['string', 'null']}}
                        }
                    }
                }
            },
            resolved_units=['rows'],
            current_page_text='Visible heading',
            previous_page_text='',
        )


def test_leaf_review_cannot_keep_text_inside_a_delimited_token() -> None:
    decision = PageLeafRepairDecision.model_validate(
        {
            'page_file': '00001.json',
            'reviews': [
                {
                    'record_index': 0,
                    'row_index': 0,
                    'field_name': 'display_value',
                    'action': 'use_source_candidate',
                    'evidence_source': 'current_page_text',
                    'evidence_line': 1,
                    'evidence_quote': 'X',
                    'rationale': 'Keep the current value.',
                }
            ],
        }
    )

    with pytest.raises(ValueError, match='line candidate'):
        validate_leaf_repair_decision(
            decision,
            page_result={
                'records': [
                    {
                        'continuation': {'is_continuation': False},
                        'rows': [{'display_value': 'X'}],
                    }
                ]
            },
            schema={
                'properties': {
                    'rows': {
                        'items': {'properties': {'display_value': {'type': 'string'}}}
                    }
                }
            },
            resolved_units=['rows'],
            current_page_text='Total (X)',
            previous_page_text='',
            allowed_targets={(0, 0, 'display_value', 'X')},
        )


def test_leaf_review_rejects_multiline_quote_without_fragment_evidence() -> None:
    joined = 'Program title Data Profiles'
    decision = PageLeafRepairDecision.model_validate(
        {
            'page_file': '00001.json',
            'reviews': [
                {
                    'record_index': 0,
                    'row_index': 0,
                    'field_name': 'program_group',
                    'action': 'use_source_candidate',
                    'evidence_source': 'current_page_text',
                    'evidence_line': 1,
                    'evidence_quote': joined,
                    'rationale': 'Keep the current value.',
                }
            ],
        }
    )

    with pytest.raises(ValueError, match='line candidate'):
        validate_leaf_repair_decision(
            decision,
            page_result={
                'records': [
                    {
                        'continuation': {'is_continuation': False},
                        'rows': [{'program_group': joined}],
                    }
                ]
            },
            schema={
                'properties': {
                    'rows': {
                        'items': {'properties': {'program_group': {'type': 'string'}}}
                    }
                }
            },
            resolved_units=['rows'],
            current_page_text='Program title\nData Profiles',
            previous_page_text='',
            allowed_targets={(0, 0, 'program_group', joined)},
        )


def test_leaf_review_accepts_an_ordered_multiline_source_span() -> None:
    joined = 'Wrapped value continues here'
    decision = PageLeafRepairDecision.model_validate(
        {
            'page_file': '00001.json',
            'reviews': [
                {
                    'record_index': 0,
                    'row_index': 0,
                    'field_name': 'description',
                    'action': 'use_source_candidate',
                    'evidence_source': 'current_page_text',
                    'evidence_line': 1,
                    'evidence_quote': 'Wrapped value',
                    'additional_evidence': [{'line': 2, 'quote': 'continues here'}],
                    'join_with': ' ',
                    'rationale': 'The value visibly wraps onto the next line.',
                }
            ],
        }
    )

    validate_leaf_repair_decision(
        decision,
        page_result={
            'records': [
                {
                    'continuation': {'is_continuation': False},
                    'rows': [{'description': joined}],
                }
            ]
        },
        schema={
            'properties': {
                'rows': {'items': {'properties': {'description': {'type': 'string'}}}}
            }
        },
        resolved_units=['rows'],
        current_page_text='Wrapped value\ncontinues here',
        previous_page_text='',
        allowed_targets={(0, 0, 'description', joined)},
    )

    assert (
        leaf_patches_from_decision(
            decision,
            page_result={
                'records': [{'rows': [{'description': joined}]}],
            },
        )
        == []
    )


def test_leaf_review_requires_evidence_for_every_target() -> None:
    decision = PageLeafRepairDecision(page_file='00001.json', reviews=[])

    with pytest.raises(ValueError, match='omitted requested targets'):
        validate_leaf_repair_decision(
            decision,
            page_result={
                'records': [
                    {
                        'continuation': {'is_continuation': False},
                        'rows': [{'label': 'Alpha'}],
                    }
                ]
            },
            schema={
                'properties': {
                    'rows': {'items': {'properties': {'label': {'type': 'string'}}}}
                }
            },
            resolved_units=['rows'],
            current_page_text='Alpha',
            previous_page_text='',
            allowed_targets={(0, 0, 'label', 'Alpha')},
        )


def test_continuation_heading_can_be_cleared_for_parser_carry() -> None:
    decision = PageLeafRepairDecision.model_validate(
        {
            'page_file': '00017.json',
            'reviews': [
                {
                    'record_index': 0,
                    'row_index': 0,
                    'field_name': 'heading',
                    'action': 'clear_for_parser_carry',
                    'evidence_source': 'previous_page_text',
                    'evidence_line': 1,
                    'evidence_quote': 'ACTIVE SECTION',
                    'rationale': 'The continuation page omits the active heading.',
                }
            ],
            'rationale': 'Clear the unsupported carried heading.',
        }
    )

    validate_leaf_repair_decision(
        decision,
        page_result={
            'records': [
                {
                    'unit_name': None,
                    'continuation': {'is_continuation': True},
                    'rows': [{'heading': 'Data row label'}],
                }
            ]
        },
        schema={
            'properties': {
                'rows': {
                    'items': {'properties': {'heading': {'type': ['string', 'null']}}}
                }
            }
        },
        resolved_units=['rows'],
        current_page_text='Data row label 10',
        previous_page_text='ACTIVE SECTION',
    )


def test_field_audit_rejects_patch_to_omitted_null_leaf() -> None:
    with pytest.raises(ValidationError, match='expected_value'):
        PageLeafRepairDecision.model_validate(
            {
                'page_file': '00001.json',
                'reviews': [
                    {
                        'record_index': 0,
                        'row_index': 0,
                        'field_name': 'heading',
                        'expected_value': None,
                        'action': 'use_source_candidate',
                        'evidence_source': 'current_page_text',
                        'evidence_line': 1,
                        'evidence_quote': 'VISIBLE HEADING',
                        'rationale': 'Fill the heading.',
                    }
                ],
            }
        )


def test_leaf_repair_prompt_contains_only_job_evidence(tmp_path) -> None:
    current_prompt = (
        tmp_path / 'agent-output' / 'longextract-unit-extract' / '00002.png_prompt.txt'
    )
    current_prompt.parent.mkdir(parents=True)
    current_prompt.write_text(
        '\nPage text:\nCURRENT PAGE\n\nPage tables:\n[]', encoding='utf-8'
    )
    previous_prompt = current_prompt.with_name('00001.png_prompt.txt')
    previous_prompt.write_text(
        '\nPage text:\nPREVIOUS PAGE\n\nPage tables:\n[]', encoding='utf-8'
    )
    table_path = tmp_path / 'agent-output' / 'tables' / '00002.json'
    table_path.parent.mkdir(parents=True)
    table_path.write_text(
        json.dumps({'rows': [{'reasoning': 'Standalone subheader'}]}),
        encoding='utf-8',
    )

    prompt = build_leaf_repair_prompt(
        tmp_path,
        page_file='00002.json',
        page_result={
            'records': [
                {
                    'continuation': {'is_continuation': False},
                    'rows': [
                        {
                            'label': 'Extracted label',
                            'optional_label': None,
                            'count': 12,
                        },
                        {
                            'label': 'Untargeted label',
                            'optional_label': 'Untargeted optional value',
                            'count': 13,
                        },
                    ],
                }
            ]
        },
        schema={
            'properties': {
                'rows': {
                    'type': 'array',
                    'items': {
                        'properties': {
                            'label': {'type': 'string'},
                            'optional_label': {'type': ['string', 'null']},
                            'count': {'type': 'integer'},
                        }
                    },
                }
            }
        },
        resolved_units=['rows'],
        field_name='label',
        expected_values=['Extracted label'],
    )

    assert str(tmp_path) not in prompt
    assert 'agent-output/longextract-unit-extract/00002.png_prompt.txt' in prompt
    assert 'agent-output/longextract-unit-extract/00001.png_prompt.txt' in prompt
    assert 'CURRENT PAGE' in prompt
    assert 'PREVIOUS PAGE' in prompt
    assert 'Standalone subheader' in prompt
    assert 'Extracted label' not in prompt
    assert 'Untargeted label' not in prompt
    assert 'Untargeted optional value' not in prompt
    assert 'Target field: label' in prompt
    assert 'Target current values:' not in prompt
    assert 'Legal target occurrences:\n' '[{"record_index":0,"row_index":0}]' in prompt
    assert 'expected_value' not in prompt
    assert 'agent-output/longextract-unit-extract/00002.json' not in prompt
    assert 'deliberately withheld' in prompt
    assert 'smallest complete source span' in prompt
    assert 'every added fragment is necessary' in prompt
    assert 'optional_label' not in prompt
    assert '"optional_label":null' not in prompt
    assert '"count":12' not in prompt
    assert 'Source artifacts:' in prompt
    assert 'L0001: CURRENT PAGE' in prompt
    assert 'L0001: PREVIOUS PAGE' in prompt
    assert 'Return exactly one review for every Legal target occurrence' in prompt
    assert 'Choose use_source_candidate' in prompt
    assert 'Do not infer or normalize symbols' in prompt


def test_section_heading_repair_follows_standalone_text_rows() -> None:
    page_result = {
        'records': [
            {
                'continuation': {'is_continuation': False},
                'rows': [
                    {'heading': 'SECTION A', 'label': 'First row'},
                    {'heading': 'SECTION A', 'label': 'Second row'},
                    {'heading': 'SECTION A', 'label': 'Third row'},
                ],
            }
        ]
    }
    schema = {
        'properties': {
            'rows': {
                'items': {
                    'properties': {
                        'heading': {
                            'type': ['string', 'null'],
                            'description': 'The nearest preceding section-only heading.',
                        },
                        'label': {'type': 'string'},
                    }
                }
            }
        }
    }

    patches, fields = infer_section_heading_patches(
        page_result,
        schema=schema,
        resolved_units=['rows'],
        current_page_text=(
            'SECTION A\n| First row | 1 |\n'
            'Subsection\n| Second row | 2 |\n| Third row | 3 |'
        ),
        previous_headings={},
    )

    assert fields == {'heading'}
    assert [patch['row_index'] for patch in patches] == [1, 2]
    assert {patch['replacement_value'] for patch in patches} == {'Subsection'}


def test_section_heading_repair_inserts_new_null_carry_transition() -> None:
    page_result = {
        'records': [
            {
                'continuation': {'is_continuation': True},
                'rows': [
                    {'heading': None, 'label': 'First row'},
                    {'heading': None, 'label': 'Second row'},
                ],
            }
        ]
    }
    schema = {
        'properties': {
            'rows': {
                'items': {
                    'properties': {
                        'heading': {
                            'type': ['string', 'null'],
                            'description': 'The nearest preceding section-only heading.',
                        },
                        'label': {'type': 'string'},
                    }
                }
            }
        }
    }

    patches, fields = infer_section_heading_patches(
        page_result,
        schema=schema,
        resolved_units=['rows'],
        current_page_text=(
            '| First row | 1 |\nNEW SECTION WITH A WRAPPED\nTITLE\n'
            '| Second row | 2 |'
        ),
        previous_headings={('rows', 'heading'): 'OLD SECTION'},
    )

    assert fields == {'heading'}
    assert patches == [
        {
            'record_index': 0,
            'row_index': 1,
            'field_name': 'heading',
            'expected_value': None,
            'replacement_value': 'NEW SECTION WITH A WRAPPED TITLE',
            'evidence_source': 'current_page_text',
            'evidence_quote': 'NEW SECTION WITH A WRAPPED TITLE',
            'rationale': (
                'Ordered page text establishes a new standalone section heading '
                'at this row.'
            ),
        }
    ]


def test_section_heading_repair_preserves_hyphenated_line_wrap() -> None:
    page_result = {
        'records': [
            {
                'continuation': {'is_continuation': False},
                'rows': [
                    {
                        'heading': 'INFLATION-ADJUSTED DOLLARS',
                        'label': 'First row',
                    }
                ],
            }
        ]
    }
    schema = {
        'properties': {
            'rows': {
                'items': {
                    'properties': {
                        'heading': {
                            'type': ['string', 'null'],
                            'description': 'The nearest preceding section-only heading.',
                        },
                        'label': {'type': 'string'},
                    }
                }
            }
        }
    }

    patches, _fields = infer_section_heading_patches(
        page_result,
        schema=schema,
        resolved_units=['rows'],
        current_page_text='INFLATION-\nADJUSTED DOLLARS\n| First row | 1 |',
        previous_headings={},
    )

    assert patches == []
