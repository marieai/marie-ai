"""Select page refinements without accepting additional row-schema violations."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from marie_longextract.ops.aggregation import aggregate_page_results


def _aggregate_errors(
    pages: list[tuple[str, dict[str, Any]]],
    validators: dict[str, Draft202012Validator],
    aggregation_policy: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    try:
        result, _trace = aggregate_page_results(pages, aggregation_policy)
    except ValueError as error:
        return True, [{'path': [], 'message': str(error)}]

    errors: list[dict[str, Any]] = []
    for unit_name, validator in validators.items():
        for row_index, row in enumerate(result.get(unit_name, [])):
            errors.extend(
                {
                    'path': [unit_name, row_index, *error.path],
                    'message': error.message,
                    'unit_name': unit_name,
                }
                for error in validator.iter_errors(row)
            )
    return False, errors


def select_refined_pages(
    initial_pages: list[tuple[str, dict[str, Any]]],
    candidate_pages: list[tuple[str, dict[str, Any]]],
    schema_units: list[dict[str, Any]],
    aggregation_policy: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    """Compare alternatives after aggregation, preserving the selected raw pages."""
    initial = dict(initial_pages)
    candidates = dict(candidate_pages)
    if (
        len(initial) != len(initial_pages)
        or len(candidates) != len(candidate_pages)
        or initial.keys() != candidates.keys()
    ):
        raise ValueError(
            'Initial and candidate page filenames must match without duplicates'
        )
    validators = {
        unit['unit_name']: Draft202012Validator(unit['schema']['items'])
        for unit in schema_units
        if unit.get('unit_kind') == 'array'
    }
    if aggregation_policy is None:
        aggregation_policy = {
            'units': {
                unit_name: {'carry_fields': [], 'sequence_fields': []}
                for unit_name in validators
            }
        }
    selected = []
    decisions = []
    for filename in sorted(initial):
        initial_failed, initial_errors = _aggregate_errors(
            [*selected, (filename, initial[filename])], validators, aggregation_policy
        )
        candidate_failed, candidate_errors = _aggregate_errors(
            [*selected, (filename, candidates[filename])],
            validators,
            aggregation_policy,
        )
        if (candidate_failed, len(candidate_errors)) > (
            initial_failed,
            len(initial_errors),
        ):
            selected.append((filename, initial[filename]))
            decisions.append(
                {
                    'filename': filename,
                    'selected': 'initial',
                    'reason': 'candidate_schema_regression',
                    'initial_error_count': len(initial_errors),
                    'candidate_error_count': len(candidate_errors),
                    'initial_error_examples': initial_errors[:5],
                    'candidate_error_examples': candidate_errors[:5],
                    'initial_aggregation_failed': initial_failed,
                    'candidate_aggregation_failed': candidate_failed,
                }
            )
        else:
            selected.append((filename, candidates[filename]))
    return selected, decisions
