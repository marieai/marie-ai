from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, TypeVar

from marie_longextract.agents.repair import (
    PageLeafRepairDecision,
    RepairDecision,
    build_leaf_repair_prompt,
    build_repair_prompt,
    leaf_patches_from_decision,
    read_prepared_page_text,
    record_patch_from_decision,
    select_leaf_repair_consensus,
    validate_decision_evidence,
    validate_leaf_repair_decision,
)
from marie_longextract.ops.aggregation import aggregate_page_results
from marie_longextract.ops.repair import (
    apply_row_leaf_patches,
    infer_section_heading_patches,
)
from pydantic import BaseModel

from marie.agent.agents import ReactAgent
from marie.agent.llm import OpenAICompatibleWrapper
from marie.agent.messages import ContentItem, Message
from marie.agent.tools.filesystem import FileReadTool
from marie.engine.output_parser import parse_json_markdown

_RAW_ANNOTATOR = 'longextract-unit-extract'
_POLICY_ANNOTATOR = 'longextract-aggregation-policy'
_DecisionT = TypeVar('_DecisionT', bound=BaseModel)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return value


def _page_results(asset_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    raw_dir = asset_dir / 'agent-output' / _RAW_ANNOTATOR
    results = [(path.name, _load_json(path)) for path in sorted(raw_dir.glob('*.json'))]
    if not results:
        raise ValueError(f'No LongExtract page results found in {raw_dir}')
    return results


def _aggregation_policy(asset_dir: Path) -> dict[str, Any]:
    policy_dir = asset_dir / 'agent-output' / _POLICY_ANNOTATOR
    paths = sorted(policy_dir.glob('*.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected one aggregation policy in {policy_dir}')
    return _load_json(paths[0])


def _active_unit_before(
    page_results: list[tuple[str, dict[str, Any]]],
    aggregation_policy: dict[str, Any],
    *,
    page_file: str,
    record_index: int,
) -> str | None:
    _result, trace = aggregate_page_results(page_results, aggregation_policy)
    decisions = [entry for entry in trace if entry['action'] != 'SUMMARY']
    for index, entry in enumerate(decisions):
        if entry['file'] == page_file and entry['record_index'] == record_index:
            return decisions[index - 1]['unit_name'] if index else None
    raise ValueError(f'No record {record_index} found in {page_file}')


def _resolved_units_for_page(
    page_results: list[tuple[str, dict[str, Any]]],
    aggregation_policy: dict[str, Any],
    *,
    page_file: str,
) -> list[str]:
    _result, trace = aggregate_page_results(page_results, aggregation_policy)
    return [
        entry['unit_name']
        for entry in trace
        if entry['action'] != 'SUMMARY' and entry['file'] == page_file
    ]


def _string_leaf_fields(
    page_result: dict[str, Any],
    schema: dict[str, Any],
    resolved_units: list[str],
) -> list[str]:
    records = page_result.get('records')
    properties = schema.get('properties')
    if not isinstance(records, list) or not isinstance(properties, dict):
        return []
    fields: set[str] = set()
    for record_index, record in enumerate(records):
        if record_index >= len(resolved_units) or not isinstance(record, dict):
            continue
        unit_schema = properties.get(resolved_units[record_index])
        item_schema = (
            unit_schema.get('items') if isinstance(unit_schema, dict) else None
        )
        field_schemas = (
            item_schema.get('properties') if isinstance(item_schema, dict) else None
        )
        rows = record.get('rows')
        if not isinstance(field_schemas, dict) or not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            for name, value in row.items():
                field_schema = field_schemas.get(name)
                if isinstance(value, str) and isinstance(field_schema, dict):
                    field_type = field_schema.get('type')
                    allowed = (
                        set(field_type)
                        if isinstance(field_type, list)
                        else {field_type}
                    )
                    if 'string' in allowed:
                        fields.add(name)
    return sorted(fields)


def _field_review_groups(
    page_result: dict[str, Any],
    field_name: str,
) -> list[list[str]]:
    records = page_result.get('records')
    if not isinstance(records, list):
        return []
    values: list[str] = []
    for record in records:
        rows = record.get('rows') if isinstance(record, dict) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            value = row.get(field_name) if isinstance(row, dict) else None
            if isinstance(value, str):
                values.append(value)
    counts = Counter(values)
    repeated = sorted(value for value, count in counts.items() if count > 1)
    singletons = sorted(value for value, count in counts.items() if count == 1)
    groups = [[value] for value in repeated]
    if singletons:
        groups.append(singletons)
    return groups


def _allowed_targets(
    page_result: dict[str, Any],
    field_name: str,
    expected_values: list[str],
) -> set[tuple[int, int, str, str]]:
    records = page_result.get('records')
    if not isinstance(records, list):
        return set()
    targets: set[tuple[int, int, str, str]] = set()
    for record_index, record in enumerate(records):
        rows = record.get('rows') if isinstance(record, dict) else None
        if not isinstance(rows, list):
            continue
        for row_index, row in enumerate(rows):
            value = row.get(field_name) if isinstance(row, dict) else None
            if isinstance(value, str) and value in expected_values:
                targets.add((record_index, row_index, field_name, value))
    return targets


def _previous_string_values(
    page_results: list[tuple[str, dict[str, Any]]],
    aggregation_policy: dict[str, Any],
    *,
    page_file: str,
) -> dict[tuple[str, str], str]:
    values: dict[tuple[str, str], str] = {}
    for filename, result in page_results:
        if filename >= page_file:
            break
        units = _resolved_units_for_page(
            page_results,
            aggregation_policy,
            page_file=filename,
        )
        records = result.get('records')
        if not isinstance(records, list):
            continue
        for record_index, record in enumerate(records):
            if record_index >= len(units) or not isinstance(record, dict):
                continue
            rows = record.get('rows')
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for field_name, value in row.items():
                    if isinstance(value, str):
                        values[(units[record_index], field_name)] = value
    return values


def _image_content(asset_dir: Path, page_number: int) -> list[ContentItem]:
    content: list[ContentItem] = []
    for label, number in (
        ('Previous page image', page_number - 1),
        ('Current page image', page_number),
        ('Next page image', page_number + 1),
    ):
        path = asset_dir / 'frames' / f'{number:05d}.png'
        if path.exists():
            content.extend((ContentItem(text=label), ContentItem(image=str(path))))
    return content


def _normalize_api_base(api_base: str) -> str:
    value = api_base.rstrip('/')
    return value if value.endswith('/v1') else f'{value}/v1'


def _collect_agent_response(agent: ReactAgent, messages: list[dict[str, Any]]) -> str:
    final_text = ''
    for responses in agent.run(messages):
        for response in responses:
            message = Message.model_validate(response)
            if (
                message.role == 'assistant'
                and message.text_content
                and not message.function_call
                and not message.tool_calls
            ):
                final_text = message.text_content
    if not final_text:
        raise ValueError('Agent returned no final response')
    return final_text


async def _run_direct_agent(
    *,
    asset_dir: Path,
    prompt: str,
    system_message: str,
    output_model: type[_DecisionT],
    page_number: int,
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int,
    max_iterations: int,
    use_file_tools: bool,
    request_timeout_seconds: float,
) -> _DecisionT:
    function_list = [FileReadTool(root_dir=asset_dir)] if use_file_tools else None
    agent = ReactAgent(
        llm=OpenAICompatibleWrapper(
            model=model,
            base_url=_normalize_api_base(api_base),
            api_key=api_key,
            tool_call_format='auto',
            timeout=request_timeout_seconds,
            max_retries=0,
        ),
        function_list=function_list,
        system_message=system_message,
        max_iterations=max_iterations,
        extra_generate_cfg={'temperature': 0.0, 'max_tokens': max_tokens},
    )
    schema = json.dumps(output_model.model_json_schema(), separators=(',', ':'))
    content = [
        ContentItem(
            text=(
                f'{prompt}\n\nReturn only one JSON object matching this JSON '
                f'Schema. Do not include commentary.\n{schema}'
            )
        ),
        *_image_content(asset_dir, page_number),
    ]
    messages = [Message.user(content).model_dump()]
    validation_error: Exception | None = None
    for _attempt in range(3):
        response_text = await asyncio.wait_for(
            asyncio.to_thread(
                _collect_agent_response,
                agent,
                messages,
            ),
            timeout=request_timeout_seconds,
        )
        try:
            payload = parse_json_markdown(response_text)
            return output_model.model_validate(payload)
        except Exception as error:
            validation_error = error
            messages.extend(
                [
                    Message.assistant(response_text).model_dump(),
                    Message.user(
                        'The previous final answer did not satisfy the required JSON '
                        f'object schema: {error}. Return a corrected JSON object only.'
                    ).model_dump(),
                ]
            )
    raise ValueError(
        'Agent did not return valid structured output'
    ) from validation_error


async def _run_agent(
    *,
    asset_dir: Path,
    page_file: str,
    record_index: int,
    active_unit: str | None,
    api_base: str,
    api_key: str,
    model: str,
    validation_feedback: str | None = None,
    request_timeout_seconds: float = 300.0,
) -> RepairDecision:
    prompt = build_repair_prompt(
        asset_dir,
        page_file=page_file,
        record_index=record_index,
        active_unit=active_unit,
    )
    if validation_feedback is not None:
        prompt += (
            '\n\nThe previous decision was rejected by the evidence validator: '
            f'{validation_feedback}. Reinspect the prepared page sequence and return '
            'a corrected decision. Do not repeat the rejected evidence claim. If no '
            'exact printed enclosing title or section marker remains, you must choose '
            'same_schema_unit.'
        )
    return await _run_direct_agent(
        asset_dir=asset_dir,
        prompt=prompt,
        system_message=(
            'You review extraction record boundaries using document evidence and '
            'filesystem tools. Use only the supplied job artifacts, never outside '
            'knowledge about standard documents. A new schema unit requires an exact '
            'printed enclosing title or section marker that is not a record value. '
            'Without one, the record remains in the active schema unit.'
        ),
        output_model=RepairDecision,
        page_number=int(Path(page_file).stem),
        api_base=api_base,
        api_key=api_key,
        model=model,
        max_tokens=2048,
        max_iterations=6,
        use_file_tools=True,
        request_timeout_seconds=request_timeout_seconds,
    )


async def _run_leaf_agent(
    *,
    asset_dir: Path,
    page_file: str,
    page_result: dict[str, Any],
    schema: dict[str, Any],
    resolved_units: list[str],
    field_name: str,
    expected_values: list[str],
    api_base: str,
    api_key: str,
    model: str,
    validation_feedback: str | None = None,
    request_timeout_seconds: float = 300.0,
) -> PageLeafRepairDecision:
    prompt = build_leaf_repair_prompt(
        asset_dir,
        page_file=page_file,
        page_result=page_result,
        schema=schema,
        resolved_units=resolved_units,
        field_name=field_name,
        expected_values=expected_values,
    )
    if validation_feedback is not None:
        prompt += (
            '\n\nReview feedback from the previous audit: '
            f'{validation_feedback}. Reinspect the cited source files and return a '
            'complete corrected review set.'
        )
    return await _run_direct_agent(
        asset_dir=asset_dir,
        prompt=prompt,
        system_message=(
            'You audit string leaves using only the supplied job files, schema, and '
            'page images. Return one evidence-backed review for every requested '
            'occurrence. The current extracted target value is withheld; select the '
            'smallest complete source value independently. Choose '
            'use_source_candidate to make an exact ordered source span the resulting '
            'value. Every fragment must match its numbered physical line. Multi-line '
            'spans require explicit ordered evidence, and every fragment must be '
            'necessary to satisfy the field schema. Preserve token punctuation. The '
            'ordered parser owns structural carry and heading transitions. Never use '
            'outside document knowledge or rewrite numeric values, row structure, '
            'record boundaries, source metadata, or sequence fields.'
        ),
        output_model=PageLeafRepairDecision,
        page_number=int(Path(page_file).stem),
        api_base=api_base,
        api_key=api_key,
        model=model,
        max_tokens=4096,
        max_iterations=8,
        use_file_tools=False,
        request_timeout_seconds=request_timeout_seconds,
    )


async def propose_boundary_repair(
    *,
    asset_dir: Path,
    page_number: int,
    record_index: int,
    api_base: str,
    api_key: str,
    model: str,
    idempotency_key: str,
    request_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    page_file = f'{page_number:05d}.json'
    results = _page_results(asset_dir)
    policy = _aggregation_policy(asset_dir)
    active_unit = _active_unit_before(
        results,
        policy,
        page_file=page_file,
        record_index=record_index,
    )
    current_page = next(value for name, value in results if name == page_file)
    current_record = current_page['records'][record_index]
    current_page_text = read_prepared_page_text(
        asset_dir
        / 'agent-output'
        / _RAW_ANNOTATOR
        / f'{page_number:05d}.png_prompt.txt'
    )
    validation_feedback: str | None = None
    for attempt in range(2):
        decision = await _run_agent(
            asset_dir=asset_dir,
            page_file=page_file,
            record_index=record_index,
            active_unit=active_unit,
            api_base=api_base,
            api_key=api_key,
            model=model,
            validation_feedback=validation_feedback,
            request_timeout_seconds=request_timeout_seconds,
        )
        if decision.page_file != page_file or decision.record_index != record_index:
            raise ValueError('Agent decision targets a different source record')
        try:
            validate_decision_evidence(
                decision,
                current_page_text=current_page_text,
                current_record=current_record,
            )
        except ValueError as error:
            if attempt == 1:
                raise
            validation_feedback = str(error)
            continue
        break
    patch = record_patch_from_decision(decision, current_record)
    action = 'patch' if patch is not None else 'keep'
    source_path = asset_dir / 'agent-output' / _RAW_ANNOTATOR / page_file
    return {
        'kind': 'boundary_repair',
        'idempotency_key': idempotency_key,
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'active_unit': active_unit,
        'decision': {
            'action': action,
            **decision.model_dump(mode='json'),
        },
        'patch': patch.model_dump(mode='json') if patch is not None else None,
    }


async def propose_leaf_repair(
    *,
    asset_dir: Path,
    page_numbers: list[int],
    schema_path: Path,
    api_base: str,
    api_key: str,
    model: str,
    idempotency_key: str,
    field_names: list[str] | None = None,
    request_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    if not page_numbers:
        raise ValueError('At least one page number is required')

    page_results = _page_results(asset_dir)
    policy = _aggregation_policy(asset_dir)
    schema = _load_json(schema_path)
    by_file = dict(page_results)

    pages: list[dict[str, Any]] = []
    requested_fields = set(field_names) if field_names is not None else None
    for page_number in sorted(set(page_numbers)):
        page_file = f'{page_number:05d}.json'
        if page_file not in by_file:
            raise ValueError(f'No page extraction found for {page_file}')
        page_result = by_file[page_file]
        resolved_units = _resolved_units_for_page(
            page_results,
            policy,
            page_file=page_file,
        )
        current_page_text = read_prepared_page_text(
            asset_dir
            / 'agent-output'
            / _RAW_ANNOTATOR
            / f'{page_number:05d}.png_prompt.txt'
        )
        previous_page_text = read_prepared_page_text(
            asset_dir
            / 'agent-output'
            / _RAW_ANNOTATOR
            / f'{page_number - 1:05d}.png_prompt.txt'
        )

        page_patches: list[dict[str, Any]] = []
        page_audits: list[dict[str, Any]] = []
        structural_patches, structural_fields = infer_section_heading_patches(
            page_result,
            schema=schema,
            resolved_units=resolved_units,
            current_page_text=current_page_text,
            previous_headings=_previous_string_values(
                page_results,
                policy,
                page_file=page_file,
            ),
        )
        if requested_fields is not None:
            structural_patches = [
                patch
                for patch in structural_patches
                if patch['field_name'] in requested_fields
            ]
        if structural_patches:
            page_result = apply_row_leaf_patches(
                page_result,
                structural_patches,
            )
            page_patches.extend(structural_patches)
        reviewed_fields = _string_leaf_fields(page_result, schema, resolved_units)
        reviewed_fields = [
            name for name in reviewed_fields if name not in structural_fields
        ]
        if requested_fields is not None:
            reviewed_fields = [
                name for name in reviewed_fields if name in requested_fields
            ]
        for field_name in reviewed_fields:
            for expected_values in _field_review_groups(page_result, field_name):
                audit_attempts: list[dict[str, Any]] = []
                valid_decisions: list[PageLeafRepairDecision] = []
                targets = _allowed_targets(
                    page_result,
                    field_name,
                    expected_values,
                )
                for opinion_index in range(3):
                    validation_feedback: str | None = None
                    for validation_attempt in range(3):
                        decision = await _run_leaf_agent(
                            asset_dir=asset_dir,
                            page_file=page_file,
                            page_result=page_result,
                            schema=schema,
                            resolved_units=resolved_units,
                            field_name=field_name,
                            expected_values=expected_values,
                            api_base=api_base,
                            api_key=api_key,
                            model=model,
                            validation_feedback=validation_feedback,
                            request_timeout_seconds=request_timeout_seconds,
                        )
                        attempt_payload: dict[str, Any] = {
                            'opinion_index': opinion_index,
                            'validation_attempt': validation_attempt,
                            'decision': decision.model_dump(mode='json'),
                        }
                        if decision.page_file != page_file:
                            raise ValueError('Agent decision targets a different page')
                        try:
                            validate_leaf_repair_decision(
                                decision,
                                page_result=page_result,
                                schema=schema,
                                resolved_units=resolved_units,
                                current_page_text=current_page_text,
                                previous_page_text=previous_page_text,
                                allowed_fields={field_name},
                                allowed_targets=targets,
                            )
                        except ValueError as error:
                            attempt_payload['validation_error'] = str(error)
                            audit_attempts.append(attempt_payload)
                            if validation_attempt == 2:
                                raise
                            validation_feedback = str(error)
                            continue
                        attempt_payload['accepted'] = True
                        audit_attempts.append(attempt_payload)
                        valid_decisions.append(decision)
                        break
                decision = select_leaf_repair_consensus(
                    page_file=page_file,
                    decisions=valid_decisions,
                    allowed_targets=targets,
                )
                validate_leaf_repair_decision(
                    decision,
                    page_result=page_result,
                    schema=schema,
                    resolved_units=resolved_units,
                    current_page_text=current_page_text,
                    previous_page_text=previous_page_text,
                    allowed_fields={field_name},
                    allowed_targets=targets,
                )
                patches = leaf_patches_from_decision(
                    decision,
                    page_result=page_result,
                )
                page_audits.append(
                    {
                        'field_name': field_name,
                        'expected_values': expected_values,
                        'attempts': audit_attempts,
                        'decision': decision.model_dump(mode='json'),
                    }
                )
                page_patches.extend(patch.model_dump(mode='json') for patch in patches)
                page_result = apply_row_leaf_patches(
                    page_result,
                    [patch.model_dump(mode='python') for patch in patches],
                )

        source_path = asset_dir / 'agent-output' / _RAW_ANNOTATOR / page_file
        payload = {
            'page_file': page_file,
            'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'patches': page_patches,
            'audits': page_audits,
            'rationale': 'Reviewed string fields: ' + ', '.join(reviewed_fields),
        }
        pages.append(payload)

    return {
        'kind': 'leaf_repair',
        'idempotency_key': idempotency_key,
        'pages': pages,
        'fields': field_names,
        'model': model,
        'patch_count': sum(len(value['patches']) for value in pages),
    }
