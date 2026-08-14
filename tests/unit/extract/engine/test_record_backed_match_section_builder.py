import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from marie.extract.engine.record_backed_match_section_builder_visitor import (
    RecordBackedMatchSectionBuilderVisitor,
)
from marie.extract.models.definition import Layer
from marie.extract.models.match import SubzeroResult


def _run_builder(
    tmp_path: Path, match_section_source: dict[str, object]
) -> SubzeroResult:
    layer = Layer(
        layer_name="layer-main",
        match_section_source={
            "strategy": "record_backed",
            "data_source": "claim-extract-aggregated",
            **match_section_source,
        },
    )
    context = SimpleNamespace(
        output_dir=tmp_path / "parsed-result",
        get_template=lambda: SimpleNamespace(layers=[layer]),
    )
    result = SubzeroResult("ROOT")

    RecordBackedMatchSectionBuilderVisitor().visit(context, result)

    return result


def test_empty_record_backed_data_source_builds_no_sections(tmp_path: Path) -> None:
    data_source_dir = tmp_path / "agent-output" / "claim-extract-aggregated"
    data_source_dir.mkdir(parents=True)
    (data_source_dir / "trace.md").write_text(
        "# Claim Aggregation Trace\n\n- Aggregated claims: 0\n",
        encoding="utf-8",
    )

    result = _run_builder(tmp_path, {})

    assert result.sections == []


def test_empty_record_backed_data_source_can_require_records(tmp_path: Path) -> None:
    data_source_dir = tmp_path / "agent-output" / "claim-extract-aggregated"
    data_source_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="but no records found"):
        _run_builder(tmp_path, {"records_required": True})


def test_missing_record_backed_data_source_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="claim-extract-aggregated"):
        _run_builder(tmp_path, {})


def test_malformed_record_backed_output_fails(tmp_path: Path) -> None:
    data_source_dir = tmp_path / "agent-output" / "claim-extract-aggregated"
    data_source_dir.mkdir(parents=True)
    (data_source_dir / "00001.json").write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        _run_builder(tmp_path, {})
