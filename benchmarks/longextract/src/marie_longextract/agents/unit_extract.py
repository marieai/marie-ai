from __future__ import annotations

import json
from typing import Any


def build_unit_task_contract(
    unit: dict[str, Any],
) -> dict[str, Any]:
    unit_name = unit["unit_name"]
    unit_kind = unit["unit_kind"]
    schema = unit["schema"]
    properties = schema.get("properties")
    if unit_kind == "array":
        properties = schema.get("items", {}).get("properties", {})
    properties = properties if isinstance(properties, dict) else {}
    required = schema.get("required")
    if unit_kind == "array":
        required = schema.get("items", {}).get("required", [])
    descriptions = {
        name: field_schema["description"]
        for name, field_schema in properties.items()
        if isinstance(field_schema, dict) and field_schema.get("description")
    }
    output_contract = (
        {unit_name: []} if unit_kind == "array" else {name: None for name in properties}
    )
    return {
        "unit_name": unit_name,
        "unit_kind": unit_kind,
        "prompt_variables": {
            "UNIT_NAME": unit_name,
            "UNIT_KIND": unit_kind,
            "UNIT_SCHEMA_JSON": json.dumps(schema, sort_keys=True),
            "REQUIRED_FIELDS_JSON": json.dumps(required or []),
            "FIELD_DESCRIPTIONS_JSON": json.dumps(descriptions, sort_keys=True),
            "OUTPUT_CONTRACT_JSON": json.dumps(output_contract, sort_keys=True),
        },
    }
