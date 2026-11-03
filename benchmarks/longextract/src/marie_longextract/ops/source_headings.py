"""Build and validate source-grounded heading requests at page boundaries."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


def heading_units(schema_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep array scope and string fields explicitly described as headings."""
    selected = []
    for unit in schema_units:
        if unit.get('unit_kind') != 'array':
            continue
        schema = unit['schema']
        fields = {}
        for name, field in schema.get('items', {}).get('properties', {}).items():
            types = field.get('type', [])
            types = [types] if isinstance(types, str) else types
            if 'string' in types and re.search(
                r'\bheadings?\b', field.get('description', ''), re.IGNORECASE
            ):
                fields[name] = field
        if fields:
            selected.append(
                {
                    'unit_name': unit['unit_name'],
                    'description': schema.get('description', ''),
                    'fields': fields,
                }
            )
    return selected


def build_source_heading_prompt(
    page_number: int,
    page_text: str,
    prior_context: Mapping[str, Any],
    units: list[dict[str, Any]],
) -> str:
    """Request incoming headings without extracting or modifying detail rows."""
    instructions = """Resolve only incoming governing headings omitted at the beginning of the
current page, for the supplied schema units and fields. Return JSON only:
{"headings": [{"unit_name": "...", "field_name": "...", "value": "...",
"source_page": 1, "line_number": 1, "source_quote": "complete printed line"}]}

Read the array description first to identify eligible detail rows and their
source-table scope. Prior table title candidates establish scope only; their
generated positions or subject matter do not supply a heading value. A new
source table must not inherit headings from an unrelated table.

For each field, inspect the first eligible current-page detail rows. Return an
incoming heading only when its governing heading is omitted before those rows.
If a qualifying heading is already printed before the first eligible row, omit
that field. Never return headings that first appear later on the current page,
headings from future pages, or values inferred from detail-row subjects. If no
eligible rows or no supported incoming headings exist, return {"headings": []}.

Read previous_page.lines backward, using each preserved source line number.
Scan backward past detail rows and household/other summary rows to the nearest
preceding heading that satisfies the field's description and still governs the
first eligible current-page rows. A heading printed after the previous page's
last detail row can govern the current page even when none of its detail rows
were printed on the previous page. Do not substitute the heading of the last
emitted detail row. Omitted middle source lines are not evidence.

Honor the schema's heading category. A geography or other group heading may
contain numeric aggregates when that is its printed role. Require a value-less
heading only when the field explicitly requires a section-only or value-less
heading. Numeric values alone neither establish nor disqualify a heading.

Every returned source_page must equal the immediately preceding page number.
Cite an existing 1-based source line on that page. source_quote must contain the
complete printed line, excluding only outer whitespace and the excerpt's added
line-number prefix. value must be a complete pipe-delimited cell or contiguous
whole whitespace tokens from that line. Preserve spelling, case, punctuation,
and delimiters; normalize repeated whitespace only. Use exact supplied unit and
field names, with at most one heading per unit/field. Omit unresolved fields;
never invent missing text or use ground truth. Do not extract detail rows or
apply a heading to all current-page rows: these values describe incoming scope
only, which ends when a new qualifying heading appears.

Source evidence (JSON):
"""
    lines = page_text.splitlines()
    opening = '\n'.join(lines[:40])[:6000]
    evidence = {
        'page_number': page_number,
        'page_text': opening,
        'prior_context': prior_context,
        'units': units,
    }
    if opening != page_text:
        evidence['page_text_truncated'] = True
        evidence['current_table_title_candidates'] = [
            {'line_number': index, 'text': line.strip()}
            for index, line in enumerate(lines, 1)
            if re.match(r'^(?:#{1,6}\s+)?table\s+[a-z0-9]', line.strip(), re.I)
        ]
    return instructions + json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
    )


def validate_source_headings(
    payload: Any,
    page_number: int,
    page_texts: Mapping[int, str],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject heading values without exact immediately preceding-page evidence."""
    if (
        not isinstance(payload, dict)
        or set(payload) != {'headings'}
        or not isinstance(payload['headings'], list)
    ):
        raise ValueError('Heading response must contain a headings array')
    allowed = {
        (unit['unit_name'], field_name)
        for unit in units
        for field_name in unit['fields']
    }
    keys = {
        'unit_name',
        'field_name',
        'value',
        'source_page',
        'line_number',
        'source_quote',
    }
    seen = set()
    validated = []
    for heading in payload['headings']:
        if not isinstance(heading, dict) or set(heading) != keys:
            raise ValueError('Heading entries must match the source-heading contract')
        unit_name = heading['unit_name']
        field_name = heading['field_name']
        if (
            not isinstance(unit_name, str)
            or not isinstance(field_name, str)
            or (unit_name, field_name) not in allowed
        ):
            raise ValueError('Unknown heading unit or field')
        identity = (unit_name, field_name)
        if identity in seen:
            raise ValueError('Duplicate heading resolution for unit and field')
        seen.add(identity)
        raw_value = heading['value']
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError('Heading value must be a non-empty string')
        value = ' '.join(raw_value.split())
        source_page = heading['source_page']
        if (
            type(source_page) is not int
            or source_page < 1
            or source_page != page_number - 1
        ):
            raise ValueError('Heading source must be the immediately preceding page')
        if source_page not in page_texts:
            raise ValueError('Heading source page is unavailable')
        lines = page_texts[source_page].splitlines()
        line_number = heading['line_number']
        if type(line_number) is not int or not 1 <= line_number <= len(lines):
            raise ValueError('Heading source line must be an existing 1-based line')
        source_line = lines[line_number - 1].strip()
        quote = heading['source_quote']
        if not isinstance(quote, str) or (
            quote.strip().removeprefix('|').removesuffix('|').strip()
            != source_line.removeprefix('|').removesuffix('|').strip()
        ):
            raise ValueError(
                'Heading quote must equal the complete printed source line'
            )
        cells = [' '.join(cell.split()) for cell in source_line.split('|')]
        tokens = source_line.split()
        value_tokens = value.split()
        if value not in cells and not any(
            tokens[index : index + len(value_tokens)] == value_tokens
            for index in range(len(tokens) - len(value_tokens) + 1)
        ):
            raise ValueError(
                'Heading value must match whole source tokens or a complete cell'
            )
        validated.append({**heading, 'value': value, 'source_quote': source_line})
    return validated
