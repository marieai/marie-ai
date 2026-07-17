from __future__ import annotations

import json
from types import SimpleNamespace

from marie_longextract.context_providers.longextract_context import (
    LongExtractContextProvider,
)

from marie.kernel import RunContext, TaskInstanceRef
from marie.kernel.backends import InMemoryStateBackend
from marie.storage import StorageManager


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
            else policy_annotation if name == "longextract-aggregation-policy" else None
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
