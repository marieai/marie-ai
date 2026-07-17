from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RecordContinuationPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')

    is_continuation: bool
    unit_name: str | None


class ContinuationBoundary(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['same_schema_unit']
    is_continuation: Literal[True]
    unit_name: None
    new_unit_marker: None


class NewSchemaUnitBoundary(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['new_schema_unit']
    is_continuation: Literal[False]
    unit_name: str = Field(min_length=1)
    new_unit_marker: str = Field(min_length=1)


SchemaBoundary = Annotated[
    ContinuationBoundary | NewSchemaUnitBoundary,
    Field(discriminator='kind'),
]


class RepairDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    page_file: str
    record_index: int = Field(ge=0)
    boundary: SchemaBoundary
    sequence_evidence: list[str] = Field(min_length=1)
    schema_evidence: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=1000)


class StringLeafPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')

    record_index: int = Field(ge=0)
    row_index: int = Field(ge=0)
    field_name: str = Field(min_length=1)
    expected_value: str
    replacement_value: str | None
    evidence_source: Literal['current_page_text', 'previous_page_text']
    evidence_quote: str = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=500)


class PageLeafRepairDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    page_file: str
    patches: list[StringLeafPatch]
    rationale: str = Field(default='', max_length=1000)


def select_leaf_repair_consensus(
    *,
    page_file: str,
    decisions: list[PageLeafRepairDecision],
    allowed_targets: set[tuple[int, int, str, str]],
) -> PageLeafRepairDecision:
    if len(decisions) < 3 or len(decisions) % 2 == 0:
        raise ValueError('Leaf repair consensus requires an odd number of audits')

    threshold = len(decisions) // 2 + 1
    selected: list[StringLeafPatch] = []
    unresolved: list[tuple[int, int, str, str]] = []
    for target in sorted(allowed_targets):
        outcomes: dict[tuple[str, str | None], int] = {}
        representatives: dict[tuple[str, str | None], StringLeafPatch] = {}
        for decision in decisions:
            if decision.page_file != page_file:
                raise ValueError('Leaf repair audit targets a different page')
            patch = next(
                (
                    candidate
                    for candidate in decision.patches
                    if (
                        candidate.record_index,
                        candidate.row_index,
                        candidate.field_name,
                        candidate.expected_value,
                    )
                    == target
                ),
                None,
            )
            outcome = (
                ('keep', None)
                if patch is None
                else ('replace', patch.replacement_value)
            )
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if patch is not None:
                representatives.setdefault(outcome, patch)

        outcome, votes = max(outcomes.items(), key=lambda item: item[1])
        if votes < threshold:
            unresolved.append(target)
            continue
        if outcome[0] == 'replace':
            selected.append(representatives[outcome])

    if unresolved:
        raise ValueError(
            'Leaf repair audits did not reach a majority for targets: ' f'{unresolved}'
        )
    return PageLeafRepairDecision(
        page_file=page_file,
        patches=selected,
        rationale=(
            f'Selected target-level majority outcomes from {len(decisions)} '
            'independent validated audits.'
        ),
    )


def _active_unit_origin(
    asset_dir: Path,
    *,
    page_file: str,
    active_unit: str | None,
) -> str | None:
    if active_unit is None:
        return None
    raw_dir = asset_dir / 'agent-output' / 'longextract-unit-extract'
    origin: str | None = None
    for path in reversed(sorted(raw_dir.glob('*.json'))):
        if path.name >= page_file:
            continue
        value = json.loads(path.read_text(encoding='utf-8'))
        records = value.get('records') if isinstance(value, dict) else None
        if not isinstance(records, list):
            continue
        for record in reversed(records):
            if not isinstance(record, dict):
                continue
            unit_name = record.get('unit_name')
            continuation = record.get('continuation')
            if (
                unit_name is None
                and isinstance(continuation, dict)
                and continuation.get('is_continuation') is True
            ):
                unit_name = active_unit
            if unit_name != active_unit:
                return origin
            origin = path.name
    return origin


def read_prepared_page_text(path: Path) -> str:
    if not path.is_file():
        return ''
    artifact = path.read_text(encoding='utf-8')
    _prefix, marker, remainder = artifact.partition('\nPage text:\n')
    if not marker:
        return ''
    page_text, marker, _remainder = remainder.partition('\n\nPage tables:\n')
    return page_text.strip() if marker else ''


def validate_decision_evidence(
    decision: RepairDecision,
    *,
    current_page_text: str,
    current_record: dict[str, object],
) -> None:
    if decision.boundary.kind != 'new_schema_unit':
        return
    marker = ' '.join(decision.boundary.new_unit_marker.split()).casefold()
    page_text = ' '.join(current_page_text.split()).casefold()
    if marker not in page_text:
        raise ValueError('new_unit_marker is not grounded in the current page text')
    record_values: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, (str, int, float)):
            record_values.add(' '.join(str(value).split()).casefold())

    collect(current_record.get('rows'))
    if marker in record_values:
        raise ValueError('new_unit_marker cannot be a value from the current record')


def record_patch_from_decision(
    decision: RepairDecision,
    current_record: dict[str, object],
) -> RecordContinuationPatch | None:
    continuation = current_record.get('continuation')
    if not isinstance(continuation, dict):
        raise ValueError('Current record continuation must be an object')
    target = RecordContinuationPatch(
        is_continuation=decision.boundary.is_continuation,
        unit_name=decision.boundary.unit_name,
    )
    if (
        continuation.get('is_continuation') == target.is_continuation
        and current_record.get('unit_name') == target.unit_name
    ):
        return None
    return target


def _normalized_text(value: str) -> str:
    return ' '.join(value.split()).casefold()


def _schema_allows(field_schema: dict[str, Any], value: str | None) -> bool:
    field_type = field_schema.get('type')
    allowed = set(field_type) if isinstance(field_type, list) else {field_type}
    return ('null' if value is None else 'string') in allowed


def validate_leaf_repair_decision(
    decision: PageLeafRepairDecision,
    *,
    page_result: dict[str, Any],
    schema: dict[str, Any],
    resolved_units: list[str],
    current_page_text: str,
    previous_page_text: str,
    allowed_fields: set[str] | None = None,
    allowed_targets: set[tuple[int, int, str, str]] | None = None,
) -> None:
    records = page_result.get('records')
    properties = schema.get('properties')
    if not isinstance(records, list) or not isinstance(properties, dict):
        raise ValueError('Page records and schema properties are required')
    if len(resolved_units) != len(records):
        raise ValueError('Resolved record units do not match page records')

    evidence = {
        'current_page_text': current_page_text,
        'previous_page_text': previous_page_text,
    }
    targets: set[tuple[int, int, str]] = set()
    for patch in decision.patches:
        if allowed_fields is not None and patch.field_name not in allowed_fields:
            raise ValueError(
                f'Patch targets {patch.field_name!r}, outside the requested field audit'
            )
        if (
            allowed_targets is not None
            and (
                patch.record_index,
                patch.row_index,
                patch.field_name,
                patch.expected_value,
            )
            not in allowed_targets
        ):
            raise ValueError('Patch target is outside the requested occurrence audit')
        target = (patch.record_index, patch.row_index, patch.field_name)
        if target in targets:
            raise ValueError(f'Duplicate leaf patch target: {target}')
        targets.add(target)

        if patch.record_index >= len(records):
            raise ValueError(f'record_index {patch.record_index} is out of range')
        record = records[patch.record_index]
        if not isinstance(record, dict):
            raise ValueError(f'records[{patch.record_index}] must be an object')
        rows = record.get('rows')
        if not isinstance(rows, list) or patch.row_index >= len(rows):
            raise ValueError(f'row_index {patch.row_index} is out of range')
        row = rows[patch.row_index]
        if not isinstance(row, dict):
            raise ValueError('Repair target row must be an object')
        if patch.field_name not in row:
            raise ValueError(
                f'Field {patch.field_name!r} is absent from the target row'
            )
        if row[patch.field_name] != patch.expected_value:
            raise ValueError(
                f'Expected value {patch.expected_value!r} does not match actual '
                f'value {row[patch.field_name]!r} at record_index '
                f'{patch.record_index}, row_index {patch.row_index}'
            )
        if patch.replacement_value == patch.expected_value:
            raise ValueError(f'Patch does not change {patch.field_name!r}')

        unit_schema = properties.get(resolved_units[patch.record_index])
        item_schema = (
            unit_schema.get('items') if isinstance(unit_schema, dict) else None
        )
        fields = (
            item_schema.get('properties') if isinstance(item_schema, dict) else None
        )
        field_schema = (
            fields.get(patch.field_name) if isinstance(fields, dict) else None
        )
        if not isinstance(field_schema, dict) or not _schema_allows(
            field_schema, patch.replacement_value
        ):
            raise ValueError(
                f'{patch.field_name!r} is not a compatible string leaf in the schema'
            )

        source_text = evidence[patch.evidence_source]
        normalized_source = _normalized_text(source_text)
        if patch.replacement_value is not None:
            replacement = _normalized_text(patch.replacement_value)
            if replacement not in normalized_source:
                raise ValueError(
                    f'Replacement for {patch.field_name!r} is not grounded in '
                    f'{patch.evidence_source}'
                )
        else:
            quote = _normalized_text(patch.evidence_quote)
            if not quote or quote not in normalized_source:
                raise ValueError(
                    f'Evidence quote for {patch.field_name!r} is not grounded in '
                    f'{patch.evidence_source}'
                )
            continuation = record.get('continuation')
            if (
                not isinstance(continuation, dict)
                or continuation.get('is_continuation') is not True
                or patch.evidence_source != 'previous_page_text'
            ):
                raise ValueError(
                    'A null replacement requires a continuation record and '
                    'previous-page evidence'
                )


def build_leaf_repair_prompt(
    asset_dir: Path,
    *,
    page_file: str,
    page_result: dict[str, Any],
    schema: dict[str, Any],
    resolved_units: list[str],
    field_name: str,
    expected_values: list[str],
) -> str:
    page_number = int(Path(page_file).stem)
    current_page_text = read_prepared_page_text(
        asset_dir
        / 'agent-output'
        / 'longextract-unit-extract'
        / f'{page_number:05d}.png_prompt.txt'
    )
    previous_page_text = read_prepared_page_text(
        asset_dir
        / 'agent-output'
        / 'longextract-unit-extract'
        / f'{page_number - 1:05d}.png_prompt.txt'
    )
    properties = schema.get('properties', {})
    target_schemas = {
        unit_name: properties[unit_name]['items']['properties'][field_name]
        for unit_name in dict.fromkeys(resolved_units)
        if isinstance(properties.get(unit_name), dict)
        and isinstance(properties[unit_name].get('items'), dict)
        and isinstance(properties[unit_name]['items'].get('properties'), dict)
        and field_name in properties[unit_name]['items']['properties']
    }
    current_path = f'agent-output/longextract-unit-extract/{page_file}'
    prompt_path = (
        'agent-output/longextract-unit-extract/' f'{page_number:05d}.png_prompt.txt'
    )
    previous_path = (
        'agent-output/longextract-unit-extract/' f'{page_number - 1:05d}.png_prompt.txt'
    )
    table_path = f'agent-output/tables/{page_file}'
    evidence_paths = [current_path, prompt_path]
    if (asset_dir / previous_path).exists():
        evidence_paths.append(previous_path)
    table_annotation: dict[str, Any] = {}
    if (asset_dir / table_path).exists():
        evidence_paths.append(table_path)
        value = json.loads((asset_dir / table_path).read_text(encoding='utf-8'))
        if isinstance(value, dict):
            table_annotation = value
    records = page_result.get('records')
    string_leaf_records: list[dict[str, Any]] = []
    target_occurrences: list[dict[str, Any]] = []
    if isinstance(records, list):
        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            unit_name = (
                resolved_units[record_index]
                if record_index < len(resolved_units)
                else None
            )
            unit_schema = properties.get(unit_name, {})
            item_schema = (
                unit_schema.get('items') if isinstance(unit_schema, dict) else {}
            )
            record_fields = (
                item_schema.get('properties') if isinstance(item_schema, dict) else {}
            )
            string_fields = {
                name
                for name, field_schema in record_fields.items()
                if isinstance(field_schema, dict) and _schema_allows(field_schema, '')
            }
            row_views: list[dict[str, Any]] = []
            rows = record.get('rows')
            if isinstance(rows, list):
                for row_index, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    target_value = row.get(field_name)
                    if not (
                        isinstance(target_value, str)
                        and target_value in expected_values
                    ):
                        continue
                    target_occurrences.append(
                        {
                            'record_index': record_index,
                            'row_index': row_index,
                            'expected_value': target_value,
                        }
                    )
                    row_views.append(
                        {
                            'row_index': row_index,
                            **{
                                name: value
                                for name, value in row.items()
                                if name in string_fields and isinstance(value, str)
                            },
                        }
                    )
            if row_views:
                string_leaf_records.append(
                    {
                        'record_index': record_index,
                        'unit_name': unit_name,
                        'continuation': record.get('continuation'),
                        'rows': row_views,
                    }
                )
    return (
        'Schema-first contract: use the field description to select the semantic '
        'value before transcribing it. Do not combine adjacent standalone headings '
        'or labels unless the description defines them as one value. Apply explicit '
        'whitespace rules only inside the selected value; whitespace normalization '
        'never expands field scope or merges separate candidates. Classify each '
        'candidate against the semantic category named by the field description. '
        'Treat complete standalone physical lines as separate candidates even when '
        'their typography matches. Join lines only when the first is grammatically '
        'incomplete and visibly wraps into the next. Every included standalone line '
        'must independently belong to the semantic category named by the field; '
        'shared typography or proximity is not sufficient. '
        'Within the selected value, printed '
        'punctuation is data: a delimited token and its undelimited interior are '
        'different strings, and interior-whitespace removal does not remove the '
        'delimiters.\n\n'
        f'Audit the {field_name!r} string field in this page extraction against the '
        'page evidence and its field schema. Return only corrections to that row '
        'field; do not add, remove, reorder, or reclassify rows and records. All '
        'required text and extraction evidence is included below.\n\n'
        f'Source artifacts:\n- ' + '\n- '.join(evidence_paths) + '\n'
        f'Current page file: {page_file}\n'
        f'Resolved unit for each record: {json.dumps(resolved_units)}\n\n'
        f'Target field: {field_name}\n'
        f'Target current values: {json.dumps(expected_values)}\n'
        f'Target field schemas by unit:\n{json.dumps(target_schemas, separators=(",", ":"))}\n\n'
        'Legal target occurrences:\n'
        f'{json.dumps(target_occurrences, separators=(",", ":"))}\n\n'
        'String leaf extraction view:\n'
        f'{json.dumps({"records": string_leaf_records}, separators=(",", ":"))}\n\n'
        f'Previous page text:\n{previous_page_text}\n\n'
        f'Current page text:\n{current_page_text}\n\n'
        'Page table annotation:\n'
        f'{json.dumps(table_annotation, separators=(",", ":"))}\n\n'
        'Patch coordinates must copy record_index and row_index exactly from the '
        'Legal target occurrences list, and expected_value must copy the value in '
        'that same entry. Those are the only legal patch targets. Both indices are '
        'zero-based array indices. Never use the document row_order or a visible row '
        'number as row_index. '
        'The extraction view intentionally contains only legal target rows and their '
        'non-null string leaves. '
        f'Audit every included occurrence of {field_name!r}, not a sample; all other '
        'fields are row context only and must not be patched. Do not patch omitted '
        'null or non-string fields. When the target value repeats, verify every '
        'occurrence and emit a separate patch for every mismatch. Apply the target '
        'field description to determine its semantic scope; do not concatenate '
        'adjacent headings or labels unless that description requires it. Printed '
        'punctuation and delimiters are part of the evidence, not optional formatting. '
        'When a field description refers to a section-only heading, a standalone '
        'text row with no data values is a heading regardless of capitalization or '
        'indentation, and the nearest such row supersedes the preceding heading. '
        'A current value is not correct merely because the same text appears somewhere '
        'on the page. Verify that it applies at the target row position using the '
        'ordered page text and the field description. '
        'For each patch, expected_value must exactly match the current JSON value. '
        'replacement_value must be the exact printed string required by the field '
        'description, including capitalization, punctuation, and delimiters. '
        'Instruction to remove whitespace inside printed delimiters does not remove '
        'the delimiters themselves. Preserve punctuation unless the field description '
        'explicitly says to remove it. Cite an exact quote from the current or '
        'previous saved prompt. Use the shortest exact contiguous excerpt that '
        'proves the replacement; do not reconstruct or reformat a table row. Use '
        'current_page_text or previous_page_text as evidence_source according to '
        'which saved prompt contains the quote. Do not infer or '
        'normalize symbols. A data-bearing label is not a section-only heading. When '
        'a continuation page omits a carried heading and the current extraction '
        'incorrectly copied a data-row label into that heading field, replace it with '
        'null and cite the active heading from previous_page_text; the ordered parser '
        'will carry it. Return an empty patches array when every string leaf is '
        'supported. Do not patch numeric values, booleans, row ordinals, unit_name, '
        'continuation, source metadata, or fields that are absent from a row.'
    )


def build_repair_prompt(
    asset_dir: Path,
    *,
    page_file: str,
    record_index: int,
    active_unit: str | None,
) -> str:
    page_stem = Path(page_file).stem
    page_number = int(page_stem)
    paths = {
        'previous extraction': f'agent-output/longextract-unit-extract/{page_number - 1:05d}.json',
        'previous extraction prompt': (
            'agent-output/longextract-unit-extract/'
            f'{page_number - 1:05d}.png_prompt.txt'
        ),
        'current extraction': f'agent-output/longextract-unit-extract/{page_file}',
        'next extraction': f'agent-output/longextract-unit-extract/{page_number + 1:05d}.json',
        'current table annotation': f'agent-output/tables/{page_file}',
        'current extraction prompt': (
            f'agent-output/longextract-unit-extract/{page_stem}.png_prompt.txt'
        ),
        'aggregation policy': 'agent-output/longextract-aggregation-policy/00001.json',
    }
    origin_file = _active_unit_origin(
        asset_dir,
        page_file=page_file,
        active_unit=active_unit,
    )
    if origin_file is not None:
        origin_stem = Path(origin_file).stem
        paths['active-unit origin extraction'] = (
            f'agent-output/longextract-unit-extract/{origin_file}'
        )
        paths['active-unit origin prompt'] = (
            'agent-output/longextract-unit-extract/' f'{origin_stem}.png_prompt.txt'
        )
    page_text_paths: list[tuple[str, Path]] = []
    if origin_file is not None and 'active-unit origin prompt' in paths:
        page_text_paths.append(
            (
                'Active-unit origin',
                asset_dir / paths['active-unit origin prompt'],
            )
        )
    page_text_paths.extend(
        [
            (
                'Previous',
                asset_dir
                / 'agent-output'
                / 'longextract-unit-extract'
                / f'{page_number - 1:05d}.png_prompt.txt',
            ),
            (
                'Current',
                asset_dir
                / 'agent-output'
                / 'longextract-unit-extract'
                / f'{page_stem}.png_prompt.txt',
            ),
        ]
    )
    page_evidence = '\n\n'.join(
        f'{label} prepared page text:\n{text}'
        for label, path in page_text_paths
        if (text := read_prepared_page_text(path))
    )
    evidence_paths = '\n'.join(
        f'- {label}: {path}'
        for label, path in paths.items()
        if (asset_dir / path).exists()
    )
    return (
        'Review one LongExtract schema-array boundary. This is not the same as a '
        'visual table boundary: repeated column headers or a newly rendered table '
        'can still continue the active schema unit.\n\n'
        f'Current page file: {page_file}\n'
        f'Current record index: {record_index}\n'
        f'Active unit before this record: {active_unit}\n\n'
        'Before deciding, use file_read to inspect the current extraction prompt, the '
        'neighboring raw extraction records, and the active-unit origin prompt when '
        'listed. Tool paths are relative to the scoped job workspace; do not prefix '
        'them with an absolute workspace path. Follow the preceding page sequence '
        'back to the title or section that '
        'establishes the active unit when the current page repeats only column headers. '
        'Saved prompts contain the schema unit descriptions and prepared page text. '
        'Compare the current rows with the descriptions of both the declared unit and '
        'the active unit. Existing unit_name and continuation values are hypotheses to '
        'verify, not evidence. The table '
        'annotation is supporting evidence about page layout, not authority over the '
        'schema-array boundary. The page images are attached in '
        'previous/current/next order.\n\n'
        f'Evidence files:\n{evidence_paths}\n\n'
        f'Prepared page sequence:\n{page_evidence}\n\n'
        'Choose same_schema_unit with is_continuation true and unit_name null when '
        'the current record belongs to the active schema unit. Choose new_schema_unit '
        'with is_continuation false and an explicit unit name when the evidence '
        'establishes a different schema '
        'unit. The runtime will determine whether that corrected boundary requires a '
        'patch. A header row or local row-order reset alone does not '
        'establish a different schema unit. Do not choose a unit from an individual '
        'row label or subject matter when multiple schema arrays share the same row '
        'shape. Schema units describe the enclosing logical profile or section. A new '
        'unit requires document evidence that establishes a different enclosing '
        'profile or section. Printed sequence markers such as 1 of N, 2 of N, and 3 '
        'of N, together with the originating title and absence of a replacement title, '
        'are evidence of one continued logical sequence even when headers repeat.\n\n'
        'For new_schema_unit, new_unit_marker must quote the exact printed profile or '
        'section title on the current page that establishes the new unit. It cannot be '
        'a data-row label, column header, existing unit_name, or inferred subject. Use '
        'null for new_unit_marker when the boundary is same_schema_unit. Use only the '
        'job artifacts: do not rely on outside knowledge about standard forms, profile '
        'codes, or where a familiar row usually appears. If the current page has no '
        'exact printed enclosing title or section marker for a new unit, choose '
        'same_schema_unit.\n\n'
        'Do not modify extracted row values. A patch may change only unit_name and '
        'continuation.is_continuation for the specified record. Cite the files or '
        'visible page evidence used in the evidence array.'
    )
