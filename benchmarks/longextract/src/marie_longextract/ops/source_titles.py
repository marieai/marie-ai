"""Resolve schema-defined title attributes with current-page source citations."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema.validators import validator_for


def title_units(schema_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select fields whose descriptions explicitly obtain values from titles."""
    selected = []
    for unit in schema_units:
        if unit.get('unit_kind') != 'array':
            continue
        schema = unit['schema']
        fields = {
            name: field
            for name, field in schema.get('items', {}).get('properties', {}).items()
            if re.search(r'\btitles?\b', field.get('description', ''), re.I)
        }
        if fields:
            selected.append(
                {
                    'unit_name': unit['unit_name'],
                    'description': schema.get('description', ''),
                    'fields': fields,
                    'required_fields': [
                        name
                        for name in schema.get('items', {}).get('required', [])
                        if name in fields
                    ],
                }
            )
    return selected


def build_source_title_prompt(
    page_number: int, page_text: str, units: list[dict[str, Any]]
) -> str:
    """Request only title attributes, separately for each printed title and unit."""
    instructions = '''Resolve only the requested schema fields supplied by table titles on this
page. Do not extract detail rows. Return JSON only:
{"titles": [{"unit_name": "...", "source_page": 1, "line_start": 1,
"line_end": 2, "source_quote": "complete title line\\nsecond title line",
"fields": {"requested_field": "source value"}}]}

Match each unit's description to its eligible printed tables. Join wrapped
title lines before separating attributes. Preserve the complete population
phrase, including descriptive group qualifiers even when they are on the
preceding title line. Do not shorten a population label to its age range alone.
Population/group qualifiers and age text belong together. A location/site is a separate attribute when the
schema requests it separately; do not append it to the population. Keep
age units with their age range even when they begin the next printed line;
age units are not part of the location. Joined text blocks below reflow only
source whitespace to make wrapped phrases readable; cite the original lines.
Extract sample counts separately as numbers. Resolve every requested attribute that
is printed, using only the field description's permitted transformations.
Do not emit null or invent an attribute that is absent from the title.

Example of a wrapped title, not actual source data: "Study - Youths\\n5-12 YR
Wrist (N=20)" supplies population "Youths 5-12 YR", site "Wrist", and count 20
when those are separate requested fields. "5-12 YR" alone drops the printed
population qualifier. Apply this full-phrase rule to the actual source below;
never copy the example values into the result.

Every returned title must include all required_fields for its unit. Keep
separately requested attributes in separate fields, not merged into one value.
Emit a separate entry for every eligible title and unit, including multiple
titles for the same unit and multiple units governed by the same title.
source_page must be the supplied current page number. line_start and line_end
are the 1-based inclusive lines containing that complete title, at most eight
lines. source_quote must reproduce those lines exactly with newlines. Every
field value must occur in that quoted source, apart from the field's requested
numeric formatting and whitespace normalization. Preserve words, case and
punctuation. Never return detail values or ground truth. If no eligible titles
are printed, return {"titles": []}.

Source and requested fields (JSON):
'''
    joined_blocks = []
    block_start = 1
    block_lines = []
    for index, line in enumerate([*page_text.splitlines(), ''], 1):
        if line.strip() and '|' not in line:
            if not block_lines:
                block_start = index
            block_lines.append(line)
        else:
            if 1 < len(block_lines) <= 8:
                joined_blocks.append(
                    {
                        'line_start': block_start,
                        'line_end': index - 1,
                        'text': ' '.join(' '.join(block_lines).split()),
                    }
                )
            block_lines = []
    return instructions + json.dumps(
        {
            'page_number': page_number,
            'lines': [
                {'line_number': index, 'text': line}
                for index, line in enumerate(page_text.splitlines(), 1)
            ],
            'units': units,
            'joined_text_blocks': joined_blocks,
        },
        ensure_ascii=False,
    )


def validate_source_titles(
    payload: Any, page_number: int, page_text: str, units: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reject attributes without a matching schema and exact current-page quote."""
    if (
        not isinstance(payload, dict)
        or set(payload) != {'titles'}
        or not isinstance(payload['titles'], list)
    ):
        raise ValueError('Title response must contain a titles array')
    allowed = {unit['unit_name']: unit['fields'] for unit in units}
    required = {
        unit['unit_name']: set(unit.get('required_fields', [])) for unit in units
    }
    lines = page_text.splitlines()
    seen = set()
    validated = []
    for title in payload['titles']:
        if not isinstance(title, dict) or set(title) != {
            'unit_name',
            'source_page',
            'line_start',
            'line_end',
            'source_quote',
            'fields',
        }:
            raise ValueError('Title entries must match the source-title contract')
        unit = title['unit_name']
        if not isinstance(unit, str) or unit not in allowed:
            raise ValueError('Unknown title unit')
        if type(title['source_page']) is not int or title['source_page'] != page_number:
            raise ValueError('Title source must be the current page')
        start, end = title['line_start'], title['line_end']
        if (
            type(start) is not int
            or type(end) is not int
            or not 1 <= start <= end <= len(lines)
            or end - start >= 8
        ):
            raise ValueError('Invalid title source line range')
        quote = '\n'.join(lines[start - 1 : end])
        if (
            not isinstance(title['source_quote'], str)
            or title['source_quote'].strip() != quote.strip()
        ):
            raise ValueError('Title quote must equal the complete source lines')
        identity = (unit, start, end)
        if identity in seen:
            raise ValueError('Duplicate title for unit and source range')
        seen.add(identity)
        fields = title['fields']
        if (
            not isinstance(fields, dict)
            or not fields
            or not set(fields) <= set(allowed[unit])
        ):
            raise ValueError('Title fields must be requested by the schema')
        if not required[unit] <= set(fields):
            raise ValueError(
                f'Missing required title attributes: {sorted(required[unit] - set(fields))}'
            )
        normalized = ' '.join(quote.split())
        for name, value in fields.items():
            field_schema = allowed[unit][name]
            if list(validator_for(field_schema)(field_schema).iter_errors(value)):
                raise ValueError(f'Title field {name} violates its schema')
            if isinstance(value, str):
                supported = bool(value.strip()) and re.search(
                    r'(?<!\w)' + re.escape(' '.join(value.split())) + r'(?!\w)',
                    normalized,
                )
            elif type(value) in (int, float):
                numbers = re.findall(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])', quote)
                supported = any(
                    float(number.replace(',', '')) == value for number in numbers
                )
            else:
                supported = False
            if not supported:
                raise ValueError(f'Title field {name} lacks exact source evidence')
        validated.append({**title, 'source_quote': quote})
    return validated
