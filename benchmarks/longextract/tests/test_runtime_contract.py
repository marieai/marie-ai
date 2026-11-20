from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from marie.kernel import RunContext, TaskInstanceRef
from marie.kernel.backends import InMemoryStateBackend
from marie_longextract.context_providers.longextract_context import (
    LongExtractContextProvider,
)

from marie.storage import StorageManager

ROOT = Path(__file__).resolve().parents[1]


def _source_context_fixture(
    annotator: str, pages: list[str]
) -> tuple[LongExtractContextProvider, SimpleNamespace]:
    run_context = SimpleNamespace(
        parameters={'extraction_units': []},
        get_annotation=lambda *_args, **_kwargs: None,
    )
    document = SimpleNamespace(
        page_count=len(pages),
        source_metadata={},
        to_text=lambda page_number: pages[page_number],
    )
    return LongExtractContextProvider(run_context, annotator), document


def test_table_context_uses_previous_source_without_replacing_page_lines() -> None:
    provider, document = _source_context_fixture(
        'tables', ['Table A. Regional counts\nREGION ALPHA | 100', 'Detail | 10']
    )

    units = provider.get_processing_units(document)
    assert [(unit.page_number, unit.index) for unit in units] == [(1, None), (2, None)]
    variables = provider.get_variables(document, 2, units[1])

    assert set(variables) == {'PREVIOUS_PAGE_CONTEXT_JSON'}
    context = json.loads(variables['PREVIOUS_PAGE_CONTEXT_JSON'])
    assert context['previous_page']['page_number'] == 1
    assert isinstance(context['previous_page']['lines'], list)
    assert '2 | REGION ALPHA | 100' in context['previous_page']['lines']
    assert context['prior_table_title_candidates'][-1] == {
        'page_number': 1,
        'line_number': 1,
        'text': 'Table A. Regional counts',
    }


def test_source_context_keeps_trailing_numeric_heading_and_table_identity() -> None:
    provider, document = _source_context_fixture(
        'tables',
        [
            'Table A. National detail',
            'continued detail',
            'Table B. Detail by region',
            '\n'.join(['detail | 1234567890'] * 500 + ['REGION OMEGA | 900 | 90']),
            'First detail | 12',
            'Table C. A future table',
        ],
    )
    provider.get_variables(document, 6)
    variables = provider.get_variables(document, 5)
    context = json.loads(variables['PREVIOUS_PAGE_CONTEXT_JSON'])

    assert len(variables['PREVIOUS_PAGE_CONTEXT_JSON']) <= 6000
    assert context['previous_page']['page_number'] == 4
    assert '501 | REGION OMEGA | 900 | 90' in context['previous_page']['lines']
    assert context['previous_page']['truncated'] is True
    assert context['prior_table_title_candidates'][-1]['text'] == (
        'Table B. Detail by region'
    )
    assert 'Table C' not in variables['PREVIOUS_PAGE_CONTEXT_JSON']


def test_first_page_source_context_does_not_borrow_future_headings() -> None:
    provider, document = _source_context_fixture(
        'tables', ['first page', 'Table B. Future region detail']
    )

    context = json.loads(
        provider.get_variables(document, 1)['PREVIOUS_PAGE_CONTEXT_JSON']
    )

    assert context == {'previous_page': None, 'prior_table_title_candidates': []}


def _heading_parameters() -> dict:
    return {
        'extraction_units': [
            {
                'prompt_variables': {
                    'UNIT_NAME': 'regional_items',
                    'UNIT_KIND': 'array',
                    'UNIT_SCHEMA_JSON': json.dumps(
                        {
                            'type': 'array',
                            'description': 'Detail rows within each regional block.',
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'region': {
                                        'type': 'string',
                                        'description': 'The governing geography heading.',
                                    },
                                    'item': {
                                        'type': 'string',
                                        'description': 'The detail label.',
                                    },
                                },
                            },
                        }
                    ),
                    'OUTPUT_CONTRACT_JSON': '{"regional_items": []}',
                }
            }
        ]
    }


def test_source_heading_context_is_limited_to_heading_schemas_and_prior_pages() -> None:
    provider, document = _source_context_fixture(
        'longextract-source-headings',
        ['Table A. Regions\n| REGION A | 100 |', 'Item one | 20', 'Item two | 30'],
    )
    provider.run_context.parameters = _heading_parameters()

    assert provider.get_eligible_pages(document) == {2, 3}
    prompt = provider.get_variables(document, 2)['SOURCE_HEADING_PROMPT']
    assert 'governing geography heading' in prompt
    assert 'REGION A' in prompt
    assert 'Item two' not in prompt
    assert 'The detail label.' not in prompt


def test_unit_context_validates_source_heading_citations() -> None:
    provider, document = _source_context_fixture(
        'longextract-unit-extract',
        ['Table A. Regions\n| REGION A | 100 |', 'Item one | 20'],
    )
    provider.run_context.parameters = _heading_parameters()
    heading = {
        'unit_name': 'regional_items',
        'field_name': 'region',
        'value': 'REGION A',
        'source_page': 1,
        'line_number': 2,
        'source_quote': '| REGION A | 100 |',
    }
    provider.run_context.get_annotation = lambda name, **kwargs: (
        {'pages': {'2': {'data': {'headings': [heading]}}}}
        if name == 'longextract-source-headings'
        else {'pages': {'1': {'data': {'units': {}}}}}
    )

    variables = provider.get_variables(document, 2)

    assert json.loads(variables['INCOMING_SOURCE_HEADINGS_JSON']) == [heading]


def test_executor_fragment_reuses_deployed_runtime_executors() -> None:
    config = yaml.safe_load((ROOT / 'config' / 'executors.yml').read_text())
    executors = {entry['name']: entry['uses'] for entry in config['executors']}

    assert set(executors) == {'annotator_llm', 'annotator_parser'}
    assert all(uses['jtype'] != 'AgentExecutor' for uses in executors.values())


def test_source_title_context_uses_current_page_only() -> None:
    provider, document = _source_context_fixture(
        'longextract-source-titles', ['Regional counts - NORTH', 'Future SOUTH']
    )
    parameters = _heading_parameters()
    contract = parameters['extraction_units'][0]['prompt_variables']
    schema = json.loads(contract['UNIT_SCHEMA_JSON'])
    schema['items']['properties']['region']['description'] = (
        'Region printed in the table title.'
    )
    contract['UNIT_SCHEMA_JSON'] = json.dumps(schema)
    provider.run_context.parameters = parameters
    assert provider.get_eligible_pages(document) == {1, 2}
    prompt = provider.get_variables(document, 1)['SOURCE_TITLE_PROMPT']
    assert 'Regional counts - NORTH' in prompt
    assert 'Future SOUTH' not in prompt


def test_unit_context_supplies_validated_title_attributes() -> None:
    provider, document = _source_context_fixture(
        'longextract-unit-extract', ['Regional counts - NORTH']
    )
    parameters = _heading_parameters()
    contract = parameters['extraction_units'][0]['prompt_variables']
    schema = json.loads(contract['UNIT_SCHEMA_JSON'])
    schema['items']['properties']['region']['description'] = (
        'Region printed in the table title.'
    )
    contract['UNIT_SCHEMA_JSON'] = json.dumps(schema)
    provider.run_context.parameters = parameters
    title = {
        'unit_name': 'regional_items',
        'source_page': 1,
        'line_start': 1,
        'line_end': 1,
        'source_quote': 'Regional counts - NORTH',
        'fields': {'region': 'NORTH'},
    }
    provider.run_context.get_annotation = lambda name, **kwargs: (
        {'pages': {'1': {'data': {'titles': [title]}}}}
        if name == 'longextract-source-titles'
        else {'pages': {'1': {'data': {'units': {}}}}}
    )
    variables = provider.get_variables(document, 1)
    assert json.loads(variables['SOURCE_TITLE_ATTRIBUTES_JSON']) == [title]


def test_context_provider_resolves_repair_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(
        StorageManager,
        "read",
        lambda _uri: json.dumps({"findings": []}).encode(),
    )
    run_context = SimpleNamespace(
        parameters={
            "prompt_variables": {"UNIT_NAME": "service_lines"},
            "prompt_variable_uris": {
                "VERIFICATION_FINDINGS_JSON": "s3://bucket/findings.json"
            },
        },
        get_annotation=lambda *_args, **_kwargs: None,
    )
    provider = LongExtractContextProvider(run_context, "longextract-repair")

    assert provider.get_variables(SimpleNamespace(), 1) == {
        "UNIT_NAME": "service_lines",
        "VERIFICATION_FINDINGS_JSON": '{"findings": []}',
    }


def test_context_provider_owns_unit_and_page_variables() -> None:
    table_annotation = {
        "pages": {
            "1": {
                "data": {
                    "extractions": [
                        {
                            "name": "Table 1",
                            "columns": ["Subject", "Estimate"],
                            "continuation": {"is_continuation": False},
                        }
                    ]
                }
            }
        }
    }
    policy_annotation = {
        "pages": {
            "1": {
                "data": {
                    "units": {
                        "service_lines": {
                            "carry_fields": [],
                            "sequence_fields": [],
                        }
                    }
                }
            }
        }
    }
    run_context = SimpleNamespace(
        parameters={
            "extraction_units": [
                {
                    "unit_name": "document_fields",
                    "unit_kind": "object",
                    "prompt_variables": {
                        "UNIT_NAME": "document_fields",
                        "UNIT_KIND": "object",
                        "UNIT_SCHEMA_JSON": '{"type": "object"}',
                        "OUTPUT_CONTRACT_JSON": '{"claim_number": null}',
                    },
                },
                {
                    "unit_name": "service_lines",
                    "unit_kind": "array",
                    "prompt_variables": {
                        "UNIT_NAME": "service_lines",
                        "UNIT_KIND": "array",
                        "UNIT_SCHEMA_JSON": '{"type": "array"}',
                        "OUTPUT_CONTRACT_JSON": '{"service_lines": []}',
                    },
                },
            ]
        },
        get_annotation=lambda name, **_kwargs: (
            table_annotation
            if name == "tables"
            else policy_annotation
            if name == "longextract-aggregation-policy"
            else None
        ),
    )
    provider = LongExtractContextProvider(run_context, "longextract-unit-extract")
    document = SimpleNamespace(
        page_count=2,
        source_metadata={"pages": "2", "ocr": ["large"]},
        to_text=lambda page_number: f"page {page_number + 1}",
    )

    units = provider.get_processing_units(document)

    assert [(unit.page_number, unit.index) for unit in units] == [(1, None), (2, None)]
    assert {unit.output_suffix for unit in units} == {""}
    variables = provider.get_variables(document, 1, units[0])
    schema_units = json.loads(variables["SCHEMA_UNITS_JSON"])
    assert [unit["unit_name"] for unit in schema_units] == [
        "document_fields",
        "service_lines",
    ]
    assert json.loads(variables["OUTPUT_CONTRACT_JSON"]) == {
        "document_fields": {"claim_number": None},
        "records": [
            {
                "unit_name": "<array unit name or null for a continuation>",
                "source": {
                    "page_index": 0,
                    "table_index": None,
                    "ocr_line_range": [],
                },
                "continuation": {
                    "is_continuation": False,
                },
                "rows": [],
            }
        ],
    }
    assert variables["PAGE_TEXT"] == "page 1"
    assert variables["DOCUMENT_METADATA_JSON"] == '{"pages": "2"}'
    assert (
        json.loads(variables["AGGREGATION_POLICY_JSON"])
        == policy_annotation["pages"]["1"]["data"]
    )
    next_page = provider.get_variables(document, 2, units[1])
    assert json.loads(next_page['PREVIOUS_PAGE_CONTEXT_JSON'])['previous_page'] == {
        'page_number': 1,
        'lines': ['1 | page 1'],
        'truncated': False,
    }


def test_policy_context_is_schema_only_and_page_one() -> None:
    run_context = SimpleNamespace(
        parameters={
            "extraction_units": [
                {
                    "prompt_variables": {
                        "UNIT_NAME": "rows",
                        "UNIT_KIND": "array",
                        "UNIT_SCHEMA_JSON": json.dumps(
                            {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"row_order": {"type": "integer"}},
                                },
                            }
                        ),
                        "OUTPUT_CONTRACT_JSON": '{"rows": []}',
                    }
                }
            ]
        },
        get_annotation=lambda *_args, **_kwargs: None,
    )
    provider = LongExtractContextProvider(run_context, "longextract-aggregation-policy")
    document = SimpleNamespace(page_count=30)

    assert provider.get_eligible_pages(document) == {1}
    variables = provider.get_variables(document, 1)
    assert json.loads(variables["SCHEMA_UNITS_JSON"])[0]["unit_name"] == "rows"
    assert json.loads(variables["POLICY_OUTPUT_CONTRACT_JSON"]) == {
        "units": {"rows": {"carry_fields": [], "sequence_fields": []}}
    }


def test_context_provider_reads_run_context_invocation_parameters() -> None:
    contract = {
        "unit_name": "document_fields",
        "prompt_variables": {
            "UNIT_NAME": "document_fields",
            "UNIT_KIND": "object",
            "UNIT_SCHEMA_JSON": '{"type": "object"}',
            "OUTPUT_CONTRACT_JSON": '{"claim_number": null}',
        },
    }
    task = TaskInstanceRef(
        tenant_id="default",
        dag_name="longextract",
        dag_id="run-1",
        task_id="longextract-unit-extract",
        try_number=1,
    )
    run_context = RunContext(
        task,
        InMemoryStateBackend(),
        parameters={"extraction_units": [contract]},
    )
    provider = LongExtractContextProvider(run_context, "longextract-unit-extract")
    document = SimpleNamespace(page_count=1)

    units = provider.get_processing_units(document)

    assert len(units) == 1
    assert units[0].index is None
    assert units[0].data is None


def test_longextract_repair_is_a_single_processing_unit() -> None:
    provider = LongExtractContextProvider(None, "longextract-repair")
    document = SimpleNamespace(page_count=40)

    assert provider.get_eligible_pages(document) == {1}
    assert [unit.page_number for unit in provider.get_processing_units(document)] == [1]
