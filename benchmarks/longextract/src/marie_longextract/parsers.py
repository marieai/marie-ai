from __future__ import annotations

import json
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from marie.extract.registry import register_parser
from marie.extract.structures import UnstructuredDocument
from marie.logging_core.predefined import default_logger as logger
from marie_longextract.ops.aggregation import aggregate_page_results
from marie_longextract.ops.refinement import select_refined_pages

_RAW_ANNOTATOR = "longextract-unit-extract"
_POLICY_ANNOTATOR = "longextract-aggregation-policy"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def _select_refinement_pages(
    raw_dir: Path, page_results: list[tuple[str, dict]], aggregation_policy: dict
) -> tuple[list[tuple[str, dict]], dict]:
    context_path = raw_dir.parent.parent / 'debug' / f'{_RAW_ANNOTATOR}_context.json'
    runs_dir = raw_dir.parent / f'.{_RAW_ANNOTATOR}-refine' / 'runs'
    if not context_path.is_file() or not runs_dir.is_dir():
        return page_results, {'status': 'no_refinement_contract', 'fallbacks': []}
    context = json.loads(context_path.read_text())
    units = context['units']
    schema_units = json.loads(units[0]['variables']['SCHEMA_UNITS_JSON'])
    if any(
        json.loads(unit['variables']['SCHEMA_UNITS_JSON']) != schema_units
        for unit in units
    ):
        raise ValueError('LongExtract page schema contracts differ within a run')

    promoted_pass = None
    success_path = raw_dir / '_SUCCESS.yaml'
    if success_path.is_file():
        success = yaml.safe_load(success_path.read_text()) or {}
        run_id = success.get('run_id')
        promoted_pass = success.get('promoted_pass')
        if (
            not isinstance(run_id, str)
            or run_id in {'.', '..'}
            or Path(run_id).name != run_id
        ):
            return page_results, {'status': 'no_refinement_identity', 'fallbacks': []}
        if (
            not isinstance(promoted_pass, str)
            or not promoted_pass.startswith('pass_')
            or not promoted_pass[5:].isdigit()
        ):
            raise ValueError('Invalid promoted refinement pass')
        if promoted_pass == 'pass_0':
            return page_results, {'status': 'initial_pass_promoted', 'fallbacks': []}
        runs = [runs_dir / run_id]
    else:
        runs = sorted(
            runs_dir.iterdir(), key=lambda path: path.stat().st_mtime_ns, reverse=True
        )
    for run in runs:
        initial_dir = run / 'pass_0'
        if not initial_dir.is_dir():
            continue
        for candidate_dir in sorted(run.glob('pass_*'), reverse=True):
            if candidate_dir == initial_dir:
                continue
            if promoted_pass is not None and candidate_dir.name != promoted_pass:
                continue
            candidate = [
                (path.name, json.loads(path.read_text()))
                for path in sorted(candidate_dir.glob('*.json'))
            ]
            if candidate != page_results:
                continue
            initial = [
                (path.name, json.loads(path.read_text()))
                for path in sorted(initial_dir.glob('*.json'))
            ]
            selected, decisions = select_refined_pages(
                initial, candidate, schema_units, aggregation_policy=aggregation_policy
            )
            return selected, {
                'status': 'checked',
                'initial_pass': str(initial_dir),
                'candidate_pass': str(candidate_dir),
                'schema_contract': str(context_path),
                'fallbacks': decisions,
            }
    return page_results, {'status': 'no_matching_refinement_pass', 'fallbacks': []}


def _write_trace(path: Path, trace: list[dict[str, object]], raw_dir: Path) -> None:
    summary = next(entry for entry in trace if entry["action"] == "SUMMARY")
    lines = [
        "# LongExtract Aggregation Trace",
        "",
        f"Source: `{raw_dir}`",
        "",
        "## Summary",
        "",
        f"- Pages: {summary['page_count']}",
        f"- Units: {summary['unit_count']}",
        f"- Rows: {summary['row_count']}",
        "",
        "## Decision Log",
        "",
    ]
    for entry in trace:
        if entry["action"] == "SUMMARY":
            continue
        lines.extend(
            [
                f"### {entry['file']} record {entry['record_index']}",
                "",
                f"- Action: {entry['action']}",
                f"- Unit: {entry['unit_name']}",
                f"- Rows: {entry['row_count']}",
                f"- Carry fields: {entry['carry_fields']}",
                f"- Sequence fields: {entry['sequence_fields']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


@register_parser("longextract-aggregated")
def parse_longextract_aggregated(
    doc: UnstructuredDocument,
    working_dir: str,
    src_dir: str,
    conf: OmegaConf,
) -> None:
    """Aggregate ordered LongExtract page records into schema-shaped JSON."""
    raw_dir = Path(src_dir).parent / _RAW_ANNOTATOR
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"LongExtract page results not found: {raw_dir}")

    page_results: list[tuple[str, dict[str, object]]] = []
    for path in sorted(raw_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            page_result = json.load(stream)
        if not isinstance(page_result, dict):
            raise ValueError(f"LongExtract page result must be an object: {path}")
        page_results.append((path.name, page_result))
    if not page_results:
        raise ValueError(f"No LongExtract page results found in {raw_dir}")

    policy_dir = raw_dir.parent / _POLICY_ANNOTATOR
    policy_files = sorted(policy_dir.glob("*.json"))
    if len(policy_files) != 1:
        raise ValueError(
            f"Expected one LongExtract aggregation policy in {policy_dir}, "
            f"found {len(policy_files)}"
        )
    with policy_files[0].open("r", encoding="utf-8") as stream:
        aggregation_policy = json.load(stream)
    if not isinstance(aggregation_policy, dict):
        raise ValueError("LongExtract aggregation policy must be an object")

    page_results, selection = _select_refinement_pages(
        raw_dir, page_results, aggregation_policy
    )
    _write_json(
        Path(working_dir) / 'debug' / 'longextract-refinement-selection.json', selection
    )

    result, trace = aggregate_page_results(
        page_results,
        aggregation_policy,
    )
    output_dir = Path(src_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.json"):
        path.unlink()

    _write_json(output_dir / "00001.json", result)
    _write_trace(output_dir / "trace.md", trace, raw_dir)
    _write_json(Path(working_dir) / "parsed-result" / "longextract-result.json", result)
    logger.info(
        f"Aggregated {len(page_results)} LongExtract page results to {output_dir}"
    )
