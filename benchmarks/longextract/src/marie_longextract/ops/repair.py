"""Deterministic reducers for applying validated LongExtract repair proposals."""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping, Sequence
from typing import Any


def _normalized_text(value: str) -> str:
    return ' '.join(value.split()).casefold()


def _clean_line(value: str) -> str:
    return value.strip().strip('#*_` ').strip()


def _join_wrapped_lines(lines: Sequence[str]) -> str:
    value = ''
    for line in lines:
        separator = '' if value.endswith('-') else ' '
        value = f'{value}{separator}{line}' if value else line
    return value


def _locate_row_anchor(
    lines: list[str],
    anchor: str,
    start_line: int,
) -> tuple[int, int] | None:
    target = _normalized_text(anchor)
    words = target.split()
    prefix = ' '.join(words[: min(4, len(words))])
    suffix = ' '.join(words[-min(2, len(words)) :])
    last_word = words[-1]
    for line_index in range(start_line, len(lines)):
        line = _normalized_text(lines[line_index])
        if target in line:
            return line_index, line_index
        if prefix not in line:
            continue
        for end_index in range(line_index + 1, min(line_index + 6, len(lines))):
            continuation = _normalized_text(lines[end_index])
            if suffix in continuation or last_word in continuation:
                return line_index, end_index
        return line_index, line_index
    return None


def _heading_candidate(
    lines: list[str],
    *,
    start_line: int,
    end_line: int,
    row_values: set[str],
    first_row_headings: set[str] | None = None,
) -> str | None:
    candidates: list[str] = []
    for raw_line in lines[start_line:end_line]:
        if '|' in raw_line:
            continue
        line = _clean_line(raw_line)
        normalized = _normalized_text(line)
        if not normalized or not any(character.isalnum() for character in line):
            continue
        if normalized in row_values:
            continue
        if first_row_headings is not None and normalized not in first_row_headings:
            continue
        candidates.append(line)
    return _join_wrapped_lines(candidates) if candidates else None


def infer_section_heading_patches(
    page_result: dict[str, Any],
    *,
    schema: dict[str, Any],
    resolved_units: Sequence[str],
    current_page_text: str,
    previous_headings: Mapping[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], set[str]]:
    records = page_result.get('records')
    properties = schema.get('properties')
    if not isinstance(records, list) or not isinstance(properties, dict):
        return [], set()

    lines = current_page_text.splitlines()
    patches: list[dict[str, Any]] = []
    handled_fields: set[str] = set()
    for record_index, record in enumerate(records):
        if record_index >= len(resolved_units) or not isinstance(record, dict):
            continue
        unit_name = resolved_units[record_index]
        unit_schema = properties.get(unit_name)
        item_schema = (
            unit_schema.get('items') if isinstance(unit_schema, dict) else None
        )
        field_schemas = (
            item_schema.get('properties') if isinstance(item_schema, dict) else None
        )
        rows = record.get('rows')
        if not isinstance(field_schemas, dict) or not isinstance(rows, list):
            continue

        heading_fields = [
            name
            for name, field_schema in field_schemas.items()
            if isinstance(field_schema, dict)
            and 'string'
            in (
                set(field_schema.get('type'))
                if isinstance(field_schema.get('type'), list)
                else {field_schema.get('type')}
            )
            and 'section-only heading'
            in str(field_schema.get('description', '')).casefold()
        ]
        for field_name in heading_fields:
            handled_fields.add(field_name)
            row_values = {
                _normalized_text(value)
                for row in rows
                if isinstance(row, dict)
                for name, value in row.items()
                if name != field_name and isinstance(value, str)
            }
            current_heading_values = {
                _normalized_text(value)
                for row in rows
                if isinstance(row, dict)
                and isinstance((value := row.get(field_name)), str)
            }
            active_heading = previous_headings.get((unit_name, field_name))
            active_source = 'previous_page_text' if active_heading else None
            continuation = record.get('continuation')
            is_continuation = (
                isinstance(continuation, dict)
                and continuation.get('is_continuation') is True
            )
            previous_end = 0
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                anchors = [
                    value
                    for name, value in row.items()
                    if name != field_name and isinstance(value, str)
                ]
                if not anchors:
                    continue
                anchor = max(anchors, key=len)
                span = _locate_row_anchor(lines, anchor, previous_end)
                if span is None:
                    continue
                candidate = _heading_candidate(
                    lines,
                    start_line=previous_end,
                    end_line=span[0],
                    row_values=row_values,
                    first_row_headings=(
                        current_heading_values if row_index == 0 else None
                    ),
                )
                previous_active_heading = active_heading
                if candidate is not None:
                    active_heading = candidate
                    active_source = 'current_page_text'

                current_value = row.get(field_name)
                if (
                    current_value is None
                    and candidate is not None
                    and (
                        previous_active_heading is None
                        or _normalized_text(previous_active_heading)
                        != _normalized_text(candidate)
                    )
                ):
                    patches.append(
                        {
                            'record_index': record_index,
                            'row_index': row_index,
                            'field_name': field_name,
                            'expected_value': None,
                            'replacement_value': candidate,
                            'evidence_source': 'current_page_text',
                            'evidence_quote': candidate,
                            'rationale': (
                                'Ordered page text establishes a new standalone '
                                'section heading at this row.'
                            ),
                        }
                    )
                if (
                    isinstance(current_value, str)
                    and isinstance(active_heading, str)
                    and _normalized_text(current_value)
                    != _normalized_text(active_heading)
                ):
                    replacement = (
                        None
                        if is_continuation and active_source == 'previous_page_text'
                        else active_heading
                    )
                    patches.append(
                        {
                            'record_index': record_index,
                            'row_index': row_index,
                            'field_name': field_name,
                            'expected_value': current_value,
                            'replacement_value': replacement,
                            'evidence_source': active_source,
                            'evidence_quote': active_heading,
                            'rationale': (
                                'Ordered page text establishes a different nearest '
                                'preceding section-only heading.'
                            ),
                        }
                    )
                previous_end = span[1] + 1
    return patches, handled_fields


def apply_record_patch(
    page_result: dict[str, Any],
    *,
    record_index: int,
    is_continuation: bool,
    unit_name: str | None,
    active_unit: str | None,
    allowed_units: Collection[str],
) -> dict[str, Any]:
    records = page_result.get('records')
    if not isinstance(records, list):
        raise ValueError('records must be an array')
    if record_index < 0 or record_index >= len(records):
        raise ValueError(f'record_index {record_index} is out of range')

    record = records[record_index]
    if not isinstance(record, dict):
        raise ValueError(f'records[{record_index}] must be an object')
    if unit_name is not None and unit_name not in allowed_units:
        raise ValueError(f'Unknown extraction unit: {unit_name}')

    if is_continuation:
        if active_unit is None and unit_name is None:
            raise ValueError('An orphan continuation must declare its unit')
        if active_unit is not None and unit_name not in (None, active_unit):
            raise ValueError(
                f"Continuation unit '{unit_name}' contradicts active unit "
                f"'{active_unit}'"
            )
    elif not unit_name:
        raise ValueError('A new parent record must declare its unit')

    current_continuation = record.get('continuation')
    if not isinstance(current_continuation, dict):
        raise ValueError(f'records[{record_index}].continuation must be an object')
    if (
        current_continuation.get('is_continuation') == is_continuation
        and record.get('unit_name') == unit_name
    ):
        raise ValueError('Repair patch does not change the source record')

    repaired = copy.deepcopy(page_result)
    repaired_record = repaired['records'][record_index]
    repaired_record['unit_name'] = unit_name
    repaired_record['continuation']['is_continuation'] = is_continuation
    return repaired


def apply_row_leaf_patches(
    page_result: dict[str, Any],
    patches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repaired = copy.deepcopy(page_result)
    records = repaired.get('records')
    if not isinstance(records, list):
        raise ValueError('records must be an array')

    for patch in patches:
        record_index = patch['record_index']
        row_index = patch['row_index']
        field_name = patch['field_name']
        record = records[record_index]
        rows = record['rows']
        row = rows[row_index]
        if row.get(field_name) != patch['expected_value']:
            raise ValueError(f'Expected value does not match {field_name!r}')
        row[field_name] = patch['replacement_value']
    return repaired
