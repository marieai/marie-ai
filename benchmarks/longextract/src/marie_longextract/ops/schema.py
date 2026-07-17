from __future__ import annotations

from typing import Any


def _is_object_array(field_schema: Any) -> bool:
    return (
        isinstance(field_schema, dict)
        and field_schema.get("type") == "array"
        and isinstance(field_schema.get("items"), dict)
        and field_schema["items"].get("type") == "object"
    )


def build_extraction_units(schema: dict[str, Any]) -> list[dict[str, Any]]:
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise ValueError("LongExtractBench schema must be a JSON object schema")

    properties = schema["properties"]
    required = set(schema.get("required") or [])
    document_properties = {
        name: value for name, value in properties.items() if not _is_object_array(value)
    }
    units: list[dict[str, Any]] = []
    if document_properties:
        document_required = [
            name
            for name in properties
            if name in required and name in document_properties
        ]
        document_schema: dict[str, Any] = {
            "type": "object",
            "properties": document_properties,
        }
        if document_required:
            document_schema["required"] = document_required
        units.append(
            {
                "unit_name": "document_fields",
                "unit_kind": "object",
                "schema": document_schema,
                "required": bool(document_required),
                "description": "Top-level document fields",
            }
        )

    for name, field_schema in properties.items():
        if not _is_object_array(field_schema):
            continue
        units.append(
            {
                "unit_name": name,
                "unit_kind": "array",
                "schema": field_schema,
                "required": name in required,
                "description": field_schema.get("description"),
            }
        )
    return units
