"""Evidence contracts and prompts for LongExtract repair agents."""

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


class SourceLineEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid')

    line: int = Field(ge=1)
    quote: str = Field(min_length=1)


class StringLeafReview(BaseModel):
    model_config = ConfigDict(extra='forbid')

    record_index: int = Field(ge=0)
    row_index: int = Field(ge=0)
    field_name: str = Field(min_length=1)
    action: Literal['use_source_candidate', 'clear_for_parser_carry']
    evidence_source: Literal['current_page_text', 'previous_page_text']
    evidence_line: int = Field(ge=1)
    evidence_quote: str = Field(min_length=1)
    additional_evidence: list[SourceLineEvidence] = Field(default_factory=list)
    join_with: Literal[' ', ''] = ' '
    rationale: str = Field(min_length=1, max_length=500)


class PageLeafRepairDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    page_file: str
    reviews: list[StringLeafReview]
    rationale: str = Field(default='', max_length=1000)


def leaf_patches_from_decision(
    decision: PageLeafRepairDecision,
    *,
    page_result: dict[str, Any],
) -> list[StringLeafPatch]:
    records = page_result.get('records')
    if not isinstance(records, list):
        raise ValueError('Page records are required to derive leaf patches')

    patches: list[StringLeafPatch] = []
    for review in decision.reviews:
        try:
            record = records[review.record_index]
            rows = record['rows']
            row = rows[review.row_index]
            expected_value = row[review.field_name]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(
                'Leaf review target is absent from the page result'
            ) from error
        if not isinstance(expected_value, str):
            raise ValueError('Leaf review target must contain a string value')

        replacement = _review_replacement(review)
        if replacement == expected_value:
            continue
        values = review.model_dump(
            exclude={'action', 'additional_evidence', 'evidence_line', 'join_with'}
        )
        values['expected_value'] = expected_value
        values['replacement_value'] = replacement
        values['evidence_quote'] = _review_quote(review)
        patches.append(StringLeafPatch.model_validate(values))
    return patches


def _review_replacement(review: StringLeafReview) -> str | None:
    if review.action == 'clear_for_parser_carry':
        return None
    return _review_quote(review)


def _review_quote(review: StringLeafReview) -> str:
    quotes = [
        review.evidence_quote,
        *(item.quote for item in review.additional_evidence),
    ]
    return review.join_with.join(quote.strip() for quote in quotes)


def select_leaf_repair_consensus(
    *,
    page_file: str,
    decisions: list[PageLeafRepairDecision],
    allowed_targets: set[tuple[int, int, str, str]],
) -> PageLeafRepairDecision:
    if len(decisions) < 3 or len(decisions) % 2 == 0:
        raise ValueError('Leaf repair consensus requires an odd number of audits')

    threshold = len(decisions) // 2 + 1
    selected: list[StringLeafReview] = []
    allowed_coordinates = {
        (record_index, row_index, field_name)
        for record_index, row_index, field_name, _expected_value in allowed_targets
    }
    if len(allowed_coordinates) != len(allowed_targets):
        raise ValueError('Leaf repair targets contain duplicate coordinates')
    unresolved: list[tuple[int, int, str]] = []
    decision_reviews: list[dict[tuple[int, int, str], StringLeafReview]] = []
    for decision in decisions:
        if decision.page_file != page_file:
            raise ValueError('Leaf repair audit targets a different page')
        reviews = {
            (
                review.record_index,
                review.row_index,
                review.field_name,
            ): review
            for review in decision.reviews
        }
        if len(reviews) != len(decision.reviews):
            raise ValueError('Leaf repair audit contains duplicate targets')
        if set(reviews) != allowed_coordinates:
            raise ValueError('Leaf repair audit must review every requested target')
        decision_reviews.append(reviews)

    for target in sorted(allowed_coordinates):
        outcomes: dict[str | None, int] = {}
        representatives: dict[str | None, StringLeafReview] = {}
        for reviews in decision_reviews:
            review = reviews[target]
            outcome = _review_replacement(review)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            representatives.setdefault(outcome, review)

        outcome, votes = max(outcomes.items(), key=lambda item: item[1])
        if votes < threshold:
            unresolved.append(target)
            continue
        selected.append(representatives[outcome])

    if unresolved:
        raise ValueError(
            'Leaf repair audits did not reach a majority for targets: ' f'{unresolved}'
        )
    return PageLeafRepairDecision(
        page_file=page_file,
        reviews=selected,
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


def _numbered_page_text(value: str) -> str:
    return '\n'.join(
        f'L{line_number:04d}: {line}'
        for line_number, line in enumerate(value.splitlines(), start=1)
    )


def _line_supports_quote(line: str, quote: str) -> bool:
    quote = quote.strip()
    if not quote:
        return False

    stripped = line.strip()
    regions = [stripped]
    if '|' in stripped:
        regions.extend(cell.strip() for cell in stripped.split('|') if cell.strip())
    for region in regions:
        if quote == region:
            return True
        source_tokens = region.split()
        quote_tokens = quote.split()
        if quote != ' '.join(quote_tokens) or len(quote_tokens) > len(source_tokens):
            continue
        width = len(quote_tokens)
        if any(
            source_tokens[offset : offset + width] == quote_tokens
            for offset in range(len(source_tokens) - width + 1)
        ):
            return True
    return False


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

    evidence_lines = {
        'current_page_text': current_page_text.splitlines(),
        'previous_page_text': previous_page_text.splitlines(),
    }
    reviewed_targets: set[tuple[int, int, str]] = set()
    allowed_values = {
        (record_index, row_index, field_name): expected_value
        for record_index, row_index, field_name, expected_value in allowed_targets
        or set()
    }
    if allowed_targets is not None and len(allowed_values) != len(allowed_targets):
        raise ValueError('Leaf repair targets contain duplicate coordinates')
    for review in decision.reviews:
        if allowed_fields is not None and review.field_name not in allowed_fields:
            raise ValueError(
                f'Review targets {review.field_name!r}, outside the requested field audit'
            )
        target = (
            review.record_index,
            review.row_index,
            review.field_name,
        )
        if allowed_targets is not None and target not in allowed_values:
            raise ValueError('Review target is outside the requested occurrence audit')
        if target in reviewed_targets:
            raise ValueError(f'Duplicate leaf review target: {target}')
        reviewed_targets.add(target)

        if review.record_index >= len(records):
            raise ValueError(f'record_index {review.record_index} is out of range')
        record = records[review.record_index]
        if not isinstance(record, dict):
            raise ValueError(f'records[{review.record_index}] must be an object')
        rows = record.get('rows')
        if not isinstance(rows, list) or review.row_index >= len(rows):
            raise ValueError(f'row_index {review.row_index} is out of range')
        row = rows[review.row_index]
        if not isinstance(row, dict):
            raise ValueError('Repair target row must be an object')
        if review.field_name not in row:
            raise ValueError(
                f'Field {review.field_name!r} is absent from the target row'
            )
        if (
            allowed_targets is not None
            and row[review.field_name] != allowed_values[target]
        ):
            raise ValueError(
                f'Runtime target value {allowed_values[target]!r} does not match '
                f'actual value {row[review.field_name]!r} at record_index '
                f'{review.record_index}, row_index {review.row_index}'
            )

        unit_schema = properties.get(resolved_units[review.record_index])
        item_schema = (
            unit_schema.get('items') if isinstance(unit_schema, dict) else None
        )
        fields = (
            item_schema.get('properties') if isinstance(item_schema, dict) else None
        )
        field_schema = (
            fields.get(review.field_name) if isinstance(fields, dict) else None
        )
        replacement = _review_replacement(review)
        if not isinstance(field_schema, dict) or not _schema_allows(
            field_schema, replacement
        ):
            raise ValueError(
                f'{review.field_name!r} is not a compatible string leaf in the schema'
            )

        source_lines = evidence_lines[review.evidence_source]
        fragments = [
            SourceLineEvidence(line=review.evidence_line, quote=review.evidence_quote),
            *review.additional_evidence,
        ]
        line_numbers = [fragment.line for fragment in fragments]
        if line_numbers != sorted(set(line_numbers)):
            raise ValueError('Source evidence lines must be unique and ordered')
        for fragment in fragments:
            if fragment.line > len(source_lines):
                raise ValueError(
                    f'Evidence line L{fragment.line:04d} is outside '
                    f'{review.evidence_source}'
                )
            source_line = source_lines[fragment.line - 1]
            if not _line_supports_quote(source_line, fragment.quote):
                raise ValueError(
                    f'Evidence quote for {review.field_name!r} is not an exact '
                    f'line candidate at {review.evidence_source} '
                    f'L{fragment.line:04d}'
                )
        if replacement is None:
            continuation = record.get('continuation')
            if (
                not isinstance(continuation, dict)
                or continuation.get('is_continuation') is not True
                or review.evidence_source != 'previous_page_text'
            ):
                raise ValueError(
                    'A null replacement requires a continuation record and '
                    'previous-page evidence'
                )
        elif review.evidence_source != 'current_page_text':
            raise ValueError(
                'A source-candidate replacement requires current-page evidence'
            )

    if allowed_targets is not None and reviewed_targets != set(allowed_values):
        missing = sorted(set(allowed_values) - reviewed_targets)
        raise ValueError(f'Leaf repair audit omitted requested targets: {missing}')


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
    prompt_path = (
        'agent-output/longextract-unit-extract/' f'{page_number:05d}.png_prompt.txt'
    )
    previous_path = (
        'agent-output/longextract-unit-extract/' f'{page_number - 1:05d}.png_prompt.txt'
    )
    table_path = f'agent-output/tables/{page_file}'
    evidence_paths = [prompt_path]
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
                        }
                    )
                    row_views.append(
                        {
                            'row_index': row_index,
                            **{
                                name: value
                                for name, value in row.items()
                                if name in string_fields
                                and name != field_name
                                and isinstance(value, str)
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
        f'Audit the {field_name!r} string field against the field schema and page '
        'evidence. Determine the smallest complete source span that satisfies the '
        'field description from its semantic meaning, row context, and printed '
        'source. Do not add, remove, reorder, or reclassify rows or records.\n\n'
        f'Source artifacts:\n- ' + '\n- '.join(evidence_paths) + '\n'
        f'Current page file: {page_file}\n'
        f'Resolved unit for each record: {json.dumps(resolved_units)}\n\n'
        f'Target field: {field_name}\n'
        'Target field schemas by unit:\n'
        f'{json.dumps(target_schemas, separators=(",", ":"))}\n\n'
        'Legal target occurrences:\n'
        f'{json.dumps(target_occurrences, separators=(",", ":"))}\n\n'
        'String leaf extraction view:\n'
        f'{json.dumps({"records": string_leaf_records}, separators=(",", ":"))}\n\n'
        'Previous page text (physical line IDs):\n'
        f'{_numbered_page_text(previous_page_text)}\n\n'
        'Current page text (physical line IDs):\n'
        f'{_numbered_page_text(current_page_text)}\n\n'
        'Page table annotation:\n'
        f'{json.dumps(table_annotation, separators=(",", ":"))}\n\n'
        'Return exactly one review for every Legal target occurrence. Copy '
        'record_index and row_index exactly from that occurrence and use only the '
        'requested field_name. Both indices are zero-based extraction array indices; '
        'a printed row number is not row_index. The current extracted target value is '
        'deliberately withheld. Determine the source value independently from the '
        'schema, neighboring row context, numbered page text, table evidence, and '
        'page image.\n\n'
        'Every review must cite at least one numbered physical line. evidence_quote '
        'and every additional_evidence quote must be either that complete line, one '
        'complete pipe-delimited table cell, or a contiguous sequence of '
        'whitespace-delimited tokens from that line. Tokens retain their printed '
        'punctuation. Do not cite a substring inside a token or construct text absent '
        'from a cited line. Use ordered additional_evidence only when every added '
        'fragment is necessary to complete the same scalar under the field schema and '
        'page structure. The primary fragment must not already be a complete value '
        'for that schema. An adjacent subtitle, category, caption, or other nested '
        'label is not part of the scalar merely because it shares style or alignment. '
        'Set join_with to the exact separator between fragments. Adjacency alone does '
        'not establish a shared value, but a value is not restricted to one physical '
        'line. Choose use_source_candidate to make the resulting smallest complete '
        'source span the value. The runtime will compare that independently selected '
        'span with the current extraction and create a patch only when they differ.\n\n'
        'Use current_page_text for values printed on this page. A null replacement '
        'requires clear_for_parser_carry and is allowed only for a continuation '
        'record whose carried value is proven by a cited previous_page_text line; '
        'the ordered parser owns that carry. Printed '
        'punctuation and delimiters are data, so a delimited token is not equivalent '
        'to its interior. The parser owns transitions between structural headings; '
        'do not turn adjacent headings into one scalar span. Do not infer or normalize '
        'symbols. Do not review numeric values, booleans, row ordinals, '
        'unit_name, continuation, source metadata, omitted fields, or other fields.'
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
