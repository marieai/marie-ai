from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, TypeVar

from marie_longextract.agents.repair import (
    PageLeafRepairDecision,
    RecordContinuationPatch,
    RepairDecision,
    build_leaf_repair_prompt,
    build_repair_prompt,
    read_prepared_page_text,
    record_patch_from_decision,
    select_leaf_repair_consensus,
    validate_decision_evidence,
    validate_leaf_repair_decision,
)
from marie_longextract.ops.repair import (
    apply_record_patch,
    apply_row_leaf_patches,
    infer_section_heading_patches,
)
from marie_longextract.ops.stitch import aggregate_page_results
from marie_longextract.parsers import parse_longextract_aggregated
from omegaconf import OmegaConf
from pydantic import BaseModel

from marie.agent.agents.assistant import ReactAgent
from marie.agent.llm_wrapper import OpenAICompatibleWrapper
from marie.agent.message import ContentItem, Message
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
        response_task = asyncio.create_task(
            asyncio.to_thread(
                _collect_agent_response,
                agent,
                messages,
            )
        )
        started = time.monotonic()
        while True:
            try:
                response_text = await asyncio.wait_for(
                    asyncio.shield(response_task),
                    timeout=30.0,
                )
                break
            except TimeoutError:
                elapsed = time.monotonic() - started
                print(
                    f'Waiting for model response for page {page_number:05d} '
                    f'(elapsed={elapsed:.0f}s, request timeout='
                    f'{request_timeout_seconds:.0f}s)',
                    flush=True,
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
    print(f'Reviewing {page_file} record {record_index} with {model}', flush=True)
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
            'corrected patch set without the unsupported change.'
        )
    print(
        f'Auditing {field_name} values {expected_values!r} in {page_file} with {model}',
        flush=True,
    )
    return await _run_direct_agent(
        asset_dir=asset_dir,
        prompt=prompt,
        system_message=(
            'You audit exact string leaves using the job files and page images. '
            'Determine the semantic span from the field description before applying '
            'transcription rules. Adjacent standalone headings are separate values '
            'unless the schema says otherwise. Whitespace normalization applies only '
            'inside the selected value and never merges separate candidates. '
            'Complete standalone physical lines remain separate candidates; join '
            'only a grammatically incomplete line that visibly wraps into the next. '
            'Every included standalone line must independently match the semantic '
            'category named by the field; shared typography is not sufficient. '
            'A punctuation-delimited token and its undelimited interior are different '
            'strings. Interior-whitespace removal never removes the delimiters. '
            'Return only exact, evidence-backed patches to existing row fields. Never '
            'use outside document knowledge or rewrite numeric values, row structure, '
            'record boundaries, source metadata, or sequence fields. Interpret schema '
            'transcription instructions literally: printed punctuation is retained '
            'unless the schema explicitly says to remove it.'
        ),
        output_model=PageLeafRepairDecision,
        page_number=int(Path(page_file).stem),
        api_base=api_base,
        api_key=api_key,
        model=model,
        max_tokens=4096,
        max_iterations=8,
        use_file_tools=True,
        request_timeout_seconds=request_timeout_seconds,
    )


def _copy_parser_inputs(asset_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    for name in (_RAW_ANNOTATOR, _POLICY_ANNOTATOR, 'tables'):
        source = asset_dir / 'agent-output' / name
        if source.exists():
            shutil.copytree(source, output_dir / 'agent-output' / name)
    frames = asset_dir / 'frames'
    if frames.exists():
        shutil.copytree(frames, output_dir / 'frames')


def _apply_and_aggregate(
    *,
    asset_dir: Path,
    output_dir: Path,
    decision: RepairDecision,
    patch: RecordContinuationPatch | None,
    active_unit: str | None,
    aggregation_policy: dict[str, Any],
) -> dict[str, Any]:
    _copy_parser_inputs(asset_dir, output_dir)
    raw_path = output_dir / 'agent-output' / _RAW_ANNOTATOR / decision.page_file
    page_result = _load_json(raw_path)

    if patch is not None:
        units = aggregation_policy.get('units')
        if not isinstance(units, dict):
            raise ValueError('Aggregation policy units must be an object')
        repaired = apply_record_patch(
            page_result,
            record_index=decision.record_index,
            is_continuation=patch.is_continuation,
            unit_name=patch.unit_name,
            active_unit=active_unit,
            allowed_units=units,
        )
        raw_path.write_text(json.dumps(repaired, indent=2), encoding='utf-8')

    repair_dir = output_dir / 'agent-output' / 'longextract-agent-repair'
    repair_dir.mkdir(parents=True)
    decision_payload = {
        'action': 'patch' if patch is not None else 'keep',
        **decision.model_dump(mode='json'),
    }
    (repair_dir / 'decision.json').write_text(
        json.dumps(decision_payload, indent=2), encoding='utf-8'
    )
    parse_longextract_aggregated(
        None,
        str(output_dir),
        str(output_dir / 'agent-output' / 'longextract-aggregated'),
        OmegaConf.create({}),
    )
    return _load_json(output_dir / 'parsed-result' / 'longextract-result.json')


async def run_repair(
    *,
    asset_dir: Path,
    output_dir: Path,
    page_number: int,
    record_index: int,
    api_base: str,
    api_key: str,
    model: str,
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
            print(f'Retrying rejected agent evidence: {error}', flush=True)
            continue
        break
    patch = record_patch_from_decision(decision, current_record)
    action = 'patch' if patch is not None else 'keep'
    print(f'Agent decision: {action} — {decision.rationale}', flush=True)

    _apply_and_aggregate(
        asset_dir=asset_dir,
        output_dir=output_dir,
        decision=decision,
        patch=patch,
        active_unit=active_unit,
        aggregation_policy=policy,
    )
    report: dict[str, Any] = {
        'asset_dir': str(asset_dir),
        'output_dir': str(output_dir),
        'active_unit': active_unit,
        'decision': {
            'action': action,
            **decision.model_dump(mode='json'),
        },
    }
    (output_dir / 'repair-evaluation.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'
    )
    print(f'Reaggregated repaired result in {output_dir}', flush=True)
    return report


async def run_leaf_repair(
    *,
    asset_dir: Path,
    output_dir: Path,
    page_numbers: list[int],
    schema_path: Path,
    api_base: str,
    api_key: str,
    model: str,
    field_names: list[str] | None = None,
    request_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    if not page_numbers:
        raise ValueError('At least one page number is required')

    page_results = _page_results(asset_dir)
    policy = _aggregation_policy(asset_dir)
    schema = _load_json(schema_path)
    by_file = dict(page_results)
    _copy_parser_inputs(asset_dir, output_dir)
    repair_dir = output_dir / 'agent-output' / 'longextract-agent-repair'
    repair_dir.mkdir(parents=True)

    decisions: list[dict[str, Any]] = []
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
            print(
                f'{page_file}: accepted {len(structural_patches)} structural '
                'heading patches',
                flush=True,
            )
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
                            print(
                                f'Retrying rejected leaf patches: {error}',
                                flush=True,
                            )
                            continue
                        attempt_payload['accepted'] = True
                        audit_attempts.append(attempt_payload)
                        valid_decisions.append(decision)
                        break
                    print(
                        f'Completed independent leaf audit {opinion_index + 1}/3',
                        flush=True,
                    )
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
                page_audits.append(
                    {
                        'field_name': field_name,
                        'expected_values': expected_values,
                        'attempts': audit_attempts,
                        'decision': decision.model_dump(mode='json'),
                    }
                )
                page_patches.extend(
                    patch.model_dump(mode='json') for patch in decision.patches
                )
                page_result = apply_row_leaf_patches(
                    page_result,
                    [patch.model_dump(mode='python') for patch in decision.patches],
                )
                print(
                    f'{page_file} {field_name} {expected_values!r}: accepted '
                    f'{len(decision.patches)} leaf patches',
                    flush=True,
                )

        raw_path = output_dir / 'agent-output' / _RAW_ANNOTATOR / page_file
        raw_path.write_text(json.dumps(page_result, indent=2), encoding='utf-8')
        payload = {
            'page_file': page_file,
            'patches': page_patches,
            'audits': page_audits,
            'rationale': 'Reviewed string fields: ' + ', '.join(reviewed_fields),
        }
        decisions.append(payload)
        (repair_dir / page_file).write_text(
            json.dumps(payload, indent=2), encoding='utf-8'
        )
        print(f'{page_file}: accepted {len(page_patches)} leaf patches', flush=True)

    parse_longextract_aggregated(
        None,
        str(output_dir),
        str(output_dir / 'agent-output' / 'longextract-aggregated'),
        OmegaConf.create({}),
    )
    report: dict[str, Any] = {
        'asset_dir': str(asset_dir),
        'output_dir': str(output_dir),
        'pages': sorted(set(page_numbers)),
        'fields': field_names,
        'model': model,
        'patch_count': sum(len(value['patches']) for value in decisions),
        'decisions': decisions,
    }
    (output_dir / 'leaf-repair-evaluation.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'
    )
    print(f'Reaggregated leaf-repaired result in {output_dir}', flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Run the LongExtract boundary repair agent against saved job assets.'
    )
    parser.add_argument('--asset-dir', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--page', required=True, type=int)
    parser.add_argument('--record-index', type=int, default=0)
    parser.add_argument(
        '--api-base',
        default=os.environ.get('LXBENCH_REPAIR_API_BASE'),
    )
    parser.add_argument(
        '--api-key',
        default=os.environ.get('LXBENCH_REPAIR_API_KEY', 'EMPTY'),
    )
    parser.add_argument('--model', default='qwen_v3_30b_instruct')
    parser.add_argument('--request-timeout-seconds', type=float, default=300.0)
    args = parser.parse_args(argv)
    if not args.api_base:
        parser.error(
            '--api-base or the LXBENCH_REPAIR_API_BASE environment variable is required'
        )
    if args.request_timeout_seconds <= 0:
        parser.error('--request-timeout-seconds must be positive')

    report = asyncio.run(
        run_repair(
            asset_dir=args.asset_dir.expanduser().resolve(),
            output_dir=args.out_dir.expanduser().resolve(),
            page_number=args.page,
            record_index=args.record_index,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
