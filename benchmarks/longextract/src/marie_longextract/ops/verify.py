from __future__ import annotations

from typing import Any

from marie_longextract.models import VerificationFinding


def verify_result(
    result: dict[str, Any],
    schema: dict[str, Any],
    attempted_units: list[dict[str, Any]],
) -> list[VerificationFinding]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Schema properties are required")

    findings: list[VerificationFinding] = []
    for unit in attempted_units:
        unit_name = unit["unit_name"]
        if unit["unit_kind"] == "array":
            value = result.get(unit_name)
            if not isinstance(value, list):
                findings.append(
                    VerificationFinding(
                        code="invalid-array",
                        unit_name=unit_name,
                        page_index=None,
                        message=f"{unit_name} must be an array",
                        repairable=True,
                    )
                )
            elif not value:
                findings.append(
                    VerificationFinding(
                        code="empty-array",
                        unit_name=unit_name,
                        page_index=None,
                        message=f"No rows were extracted for attempted unit {unit_name}",
                        repairable=True,
                    )
                )
            continue

        for field_name in unit["schema"].get("required", []):
            if field_name not in result:
                findings.append(
                    VerificationFinding(
                        code="missing-required-field",
                        unit_name=unit_name,
                        page_index=None,
                        message=f"Required field {field_name} was not extracted",
                        repairable=True,
                    )
                )
    return findings
