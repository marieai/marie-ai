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


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be an array of strings")
    return value


def _page_index(filename: str) -> int:
    page_name = filename.removesuffix(".json").split("_", 1)[0]
    try:
        return int(page_name) - 1
    except ValueError as error:
        raise ValueError(f"Invalid page result filename: {filename}") from error


def _policy_by_unit(
    aggregation_policy: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    units = aggregation_policy.get("units")
    if not isinstance(units, dict):
        raise ValueError("aggregation policy units must be an object")

    resolved: dict[str, dict[str, list[str]]] = {}
    for unit_name, value in units.items():
        if not isinstance(unit_name, str) or not unit_name:
            raise ValueError("aggregation policy unit names must be strings")
        if not isinstance(value, dict):
            raise ValueError(f"aggregation policy for {unit_name} must be an object")
        resolved[unit_name] = {
            "carry_fields": _string_list(
                value.get("carry_fields"), f"{unit_name}.carry_fields"
            ),
            "sequence_fields": _string_list(
                value.get("sequence_fields"), f"{unit_name}.sequence_fields"
            ),
        }
    return resolved


def aggregate_page_results(
    page_results: list[tuple[str, dict[str, Any]]],
    aggregation_policy: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result: dict[str, Any] = {}
    rows_by_unit: dict[str, list[dict[str, Any]]] = {}
    carry_state: dict[str, dict[str, Any]] = {}
    policy_by_unit = _policy_by_unit(aggregation_policy)
    trace: list[dict[str, Any]] = []
    active_unit: str | None = None

    for filename, page_result in sorted(page_results, key=lambda item: item[0]):
        page_index = _page_index(filename)
        document_fields = page_result.get("document_fields", {})
        if not isinstance(document_fields, dict):
            raise ValueError(f"document_fields must be an object in {filename}")
        for name, value in document_fields.items():
            if name not in result and value not in ("", None):
                result[name] = copy.deepcopy(value)

        records = page_result.get("records")
        if not isinstance(records, list):
            raise ValueError(f"records must be an array in {filename}")

        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"records[{record_index}] must be an object in {filename}"
                )

            continuation = record.get("continuation")
            if not isinstance(continuation, dict):
                raise ValueError(
                    f"records[{record_index}].continuation must be an object in {filename}"
                )
            is_continuation = continuation.get("is_continuation")
            if not isinstance(is_continuation, bool):
                raise ValueError(
                    f"records[{record_index}].continuation.is_continuation "
                    f"must be a boolean in {filename}"
                )

            source = record.get("source", {})
            if not isinstance(source, dict):
                raise ValueError(
                    f"records[{record_index}].source must be an object in {filename}"
                )
            declared_unit = record.get("unit_name")
            if declared_unit is not None and not isinstance(declared_unit, str):
                raise ValueError(
                    f"records[{record_index}].unit_name must be a string or null in {filename}"
                )

            action = "NEW_PARENT"
            if is_continuation:
                if active_unit is None:
                    if declared_unit is None:
                        raise ValueError(
                            f"Orphan continuation at records[{record_index}] in {filename}"
                        )
                    active_unit = declared_unit
                    action = "ORPHAN"
                elif declared_unit is not None and declared_unit != active_unit:
                    raise ValueError(
                        f"Continuation unit '{declared_unit}' contradicts active unit "
                        f"'{active_unit}' in {filename}"
                    )
                else:
                    action = "MERGE"
                unit_name = active_unit
            else:
                if not declared_unit:
                    raise ValueError(
                        f"New record at records[{record_index}] requires unit_name in {filename}"
                    )
                unit_name = declared_unit
                active_unit = unit_name
                carry_state[unit_name] = {}

            unit_policy = policy_by_unit.get(unit_name)
            if unit_policy is None:
                raise ValueError(
                    f"Aggregation policy is missing array unit '{unit_name}'"
                )
            carry_fields = unit_policy["carry_fields"]
            sequence_fields = unit_policy["sequence_fields"]

            rows = record.get("rows")
            if not isinstance(rows, list) or not all(
                isinstance(row, dict) for row in rows
            ):
                raise ValueError(
                    f"records[{record_index}].rows must be an array of objects in {filename}"
                )

            state = carry_state.setdefault(unit_name, {})
            aggregated_rows = rows_by_unit.setdefault(unit_name, [])
            for source_row in rows:
                row = copy.deepcopy(source_row)
                if is_continuation:
                    for field_name in carry_fields:
                        if row.get(field_name) in ("", None) and field_name in state:
                            row[field_name] = copy.deepcopy(state[field_name])
                for field_name, value in row.items():
                    if value not in ("", None):
                        state[field_name] = copy.deepcopy(value)
                aggregated_rows.append(row)

            normalized_source = copy.deepcopy(source)
            normalized_source["page_index"] = page_index
            trace.append(
                {
                    "action": action,
                    "file": filename,
                    "record_index": record_index,
                    "unit_name": unit_name,
                    "source": normalized_source,
                    "row_count": len(rows),
                    "carry_fields": carry_fields,
                    "sequence_fields": sequence_fields,
                }
            )

    for unit_name, rows in rows_by_unit.items():
        unit_policy = policy_by_unit[unit_name]
        policy_fields = {
            *unit_policy["carry_fields"],
            *unit_policy["sequence_fields"],
        }
        missing_fields = sorted(
            field_name
            for field_name in policy_fields
            if rows and not any(field_name in row for row in rows)
        )
        if missing_fields:
            raise ValueError(
                f"Aggregation policy for '{unit_name}' references fields absent "
                f"from extracted rows: {missing_fields}"
            )
        for row_number, row in enumerate(rows, 1):
            for field_name in unit_policy["sequence_fields"]:
                row[field_name] = row_number
        result[unit_name] = rows

    trace.append(
        {
            "action": "SUMMARY",
            "page_count": len(page_results),
            "unit_count": len(rows_by_unit),
            "row_count": sum(len(rows) for rows in rows_by_unit.values()),
        }
    )
    return result, trace
