from __future__ import annotations

import json
from pathlib import Path

from marie_longextract.ops.aggregation import aggregate_page_results
from omegaconf import OmegaConf

from marie.extract.registry import register_parser
from marie.extract.structures import UnstructuredDocument
from marie.logging_core.predefined import default_logger as logger

_RAW_ANNOTATOR = "longextract-unit-extract"
_POLICY_ANNOTATOR = "longextract-aggregation-policy"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


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
