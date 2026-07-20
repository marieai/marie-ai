from __future__ import annotations

import copy
import json
from typing import Any

from marie_longextract.models import PartialExtraction


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    return value


def _row_signature(row: dict[str, Any]) -> str:
    return json.dumps(_normalize(row), sort_keys=True, separators=(",", ":"))


def _allows_null(field_schema: Any) -> bool:
    if not isinstance(field_schema, dict):
        return False
    field_type = field_schema.get("type")
    if field_type == "null" or (isinstance(field_type, list) and "null" in field_type):
        return True
    return any(
        isinstance(option, dict) and option.get("type") == "null"
        for option in field_schema.get("anyOf", [])
    )


def stitch_partials(
    partials: list[PartialExtraction],
    schema: dict[str, Any],
) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Schema properties are required")

    result: dict[str, Any] = {}
    attempted_units = {partial.unit_name for partial in partials}
    document_partials = [
        partial for partial in partials if partial.unit_name == "document_fields"
    ]
    for partial in sorted(document_partials, key=lambda value: value.page_index):
        for name, value in partial.scalars.items():
            if name in properties and name not in result and value not in ("", None):
                result[name] = value

    if "document_fields" in attempted_units:
        for name, field_schema in properties.items():
            if name not in result and _allows_null(field_schema):
                result[name] = None

    for unit_name in sorted(attempted_units - {"document_fields"}):
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        unit_partials = sorted(
            (partial for partial in partials if partial.unit_name == unit_name),
            key=lambda value: value.page_index,
        )
        for partial in unit_partials:
            for row in partial.rows:
                signature = _row_signature(row)
                if signature not in seen:
                    seen.add(signature)
                    rows.append(row)
        result[unit_name] = rows
    return result
