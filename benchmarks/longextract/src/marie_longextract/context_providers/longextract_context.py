from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from marie.extract.annotators.context_provider import ProcessingUnit
from marie.extract.annotators.context_providers.table_context import (
    TableContextProvider,
)
from marie.extract.registry import register_context_provider
from marie.storage import StorageManager

if TYPE_CHECKING:
    from marie.extract.structures.unstructured_document import UnstructuredDocument
    from marie.kernel.context import RunContext

_UNIT_ANNOTATOR = "longextract-unit-extract"
_POLICY_ANNOTATOR = "longextract-aggregation-policy"
_REPAIR_ANNOTATOR = "longextract-repair"


@register_context_provider(
    name="longextract",
    target_annotators=[_POLICY_ANNOTATOR, _UNIT_ANNOTATOR, _REPAIR_ANNOTATOR],
)
class LongExtractContextProvider(TableContextProvider):
    """Provide planner context and processing units for LongExtract annotators."""

    def __init__(
        self,
        run_context: "RunContext | None",
        annotator_name: str,
        mode: str = "per-page",
    ) -> None:
        super().__init__(run_context, annotator_name, mode="per-page")
        self._resolved_repair_contract: dict[str, str] | None = None
        self._resolved_page_contract: dict[str, str] | None = None
        self._resolved_policy_contract: dict[str, str] | None = None

    def _parameters(self) -> Mapping[str, Any]:
        if self.run_context is None:
            return {}
        return self.run_context.parameters

    def _resolve_variables(self, contract: Mapping[str, Any]) -> dict[str, str]:
        variables = contract.get("prompt_variables") or {}
        if not isinstance(variables, dict):
            raise ValueError("prompt_variables must be an object")
        resolved = {
            str(key): (
                value if isinstance(value, str) else json.dumps(value, sort_keys=True)
            )
            for key, value in variables.items()
        }

        variable_uris = contract.get("prompt_variable_uris") or {}
        if not isinstance(variable_uris, dict):
            raise ValueError("prompt_variable_uris must be an object")
        for key, uri in variable_uris.items():
            if not isinstance(uri, str) or not uri:
                raise ValueError(f"Prompt variable URI for {key} must be a string")
            raw = StorageManager.read(uri)
            value = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            resolved[str(key)] = json.dumps(value, sort_keys=True)
        return resolved

    def _repair_variables(self) -> dict[str, str]:
        if self._resolved_repair_contract is None:
            self._resolved_repair_contract = self._resolve_variables(self._parameters())
        return dict(self._resolved_repair_contract)

    def _unit_contracts(self) -> list[dict[str, Any]]:
        contracts = self._parameters().get("extraction_units")
        if not isinstance(contracts, list) or not contracts:
            raise ValueError("extraction_units must be a non-empty array")
        if not all(isinstance(contract, dict) for contract in contracts):
            raise ValueError("Every extraction unit must be an object")
        return contracts

    def _page_variables(self) -> dict[str, str]:
        if self._resolved_page_contract is not None:
            return dict(self._resolved_page_contract)

        units = []
        document_fields: dict[str, Any] = {}
        for contract in self._unit_contracts():
            variables = self._resolve_variables(contract)
            unit_name = variables["UNIT_NAME"]
            unit_kind = variables["UNIT_KIND"]
            units.append(
                {
                    "unit_name": unit_name,
                    "unit_kind": unit_kind,
                    "schema": json.loads(variables["UNIT_SCHEMA_JSON"]),
                }
            )
            if unit_kind == "object":
                document_fields.update(json.loads(variables["OUTPUT_CONTRACT_JSON"]))

        output_contract = {
            "document_fields": document_fields,
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

        self._resolved_page_contract = {
            "SCHEMA_UNITS_JSON": json.dumps(units, sort_keys=True),
            "OUTPUT_CONTRACT_JSON": json.dumps(output_contract, sort_keys=True),
        }
        return dict(self._resolved_page_contract)

    def _policy_variables(self) -> dict[str, str]:
        if self._resolved_policy_contract is not None:
            return dict(self._resolved_policy_contract)

        units = []
        policy_units: dict[str, dict[str, list[str]]] = {}
        for contract in self._unit_contracts():
            variables = self._resolve_variables(contract)
            if variables["UNIT_KIND"] != "array":
                continue
            unit_name = variables["UNIT_NAME"]
            units.append(
                {
                    "unit_name": unit_name,
                    "schema": json.loads(variables["UNIT_SCHEMA_JSON"]),
                }
            )
            policy_units[unit_name] = {
                "carry_fields": [],
                "sequence_fields": [],
            }

        self._resolved_policy_contract = {
            "SCHEMA_UNITS_JSON": json.dumps(units, sort_keys=True),
            "POLICY_OUTPUT_CONTRACT_JSON": json.dumps(
                {"units": policy_units}, sort_keys=True
            ),
        }
        return dict(self._resolved_policy_contract)

    def _aggregation_policy(self) -> dict[str, Any]:
        if self.run_context is None:
            raise ValueError(
                "RunContext is required for LongExtract aggregation policy"
            )
        annotation = self.run_context.get_annotation(_POLICY_ANNOTATOR)
        pages = annotation.get("pages", {}) if isinstance(annotation, dict) else {}
        for page_number in sorted(pages, key=int):
            page = pages[page_number]
            data = page.get("data") if isinstance(page, dict) else None
            if isinstance(data, dict):
                return data
        raise ValueError("LongExtract aggregation policy annotation is missing")

    def get_eligible_pages(self, document: "UnstructuredDocument") -> set[int]:
        if document.page_count < 1:
            return set()
        if self.annotator_name in {_POLICY_ANNOTATOR, _REPAIR_ANNOTATOR}:
            return {1}
        return set(range(1, document.page_count + 1))

    def get_processing_units(
        self, document: "UnstructuredDocument"
    ) -> list[ProcessingUnit]:
        if self.annotator_name in {_POLICY_ANNOTATOR, _REPAIR_ANNOTATOR}:
            return [
                ProcessingUnit(page_number=page_number)
                for page_number in sorted(self.get_eligible_pages(document))
            ]

        self._unit_contracts()
        return [
            ProcessingUnit(page_number=page_number, index=None)
            for page_number in sorted(self.get_eligible_pages(document))
        ]

    def get_variables(
        self,
        document: "UnstructuredDocument",
        page_number: int,
        unit: ProcessingUnit | None = None,
    ) -> dict[str, str]:
        if self.annotator_name == _POLICY_ANNOTATOR:
            return self._policy_variables()
        if self.annotator_name == _REPAIR_ANNOTATOR:
            return self._repair_variables()
        variables = self._page_variables()

        source_metadata = document.source_metadata
        document_metadata = (
            {key: value for key, value in source_metadata.items() if key != "ocr"}
            if isinstance(source_metadata, dict)
            else {}
        )
        variables.update(
            {
                "PAGE_NUMBER": str(page_number),
                "PAGE_TEXT": document.to_text(page_number=page_number - 1),
                "PAGE_TABLES_JSON": json.dumps(
                    self._tables_by_page.get(page_number, []),
                    sort_keys=True,
                    default=str,
                ),
                "AGGREGATION_POLICY_JSON": json.dumps(
                    self._aggregation_policy(), sort_keys=True
                ),
                "DOCUMENT_METADATA_JSON": json.dumps(
                    document_metadata, sort_keys=True, default=str
                ),
            }
        )
        return variables
