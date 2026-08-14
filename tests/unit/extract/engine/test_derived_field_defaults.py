from types import SimpleNamespace

import pytest

from marie.extract.engine.match_section_extract_visitor import (
    MatchSectionExtractionProcessingVisitor,
)
from marie.extract.engine.record_backed_match_section_population_visitor import (
    RecordBackedMatchSectionPopulationVisitor,
    _create_fields,
)
from marie.extract.models.match import Field, MatchSection
from marie.extract.structures.line_with_meta import LineWithMeta


def _build_fields(
    record_backed: bool,
    transformed_value: dict[str, str | None],
    derived_field: str | dict[str, object] | None = None,
) -> list[Field]:
    if derived_field is None:
        derived_field = {
            "name": "PROCEDURE_CODE",
            "type": "ALPHA_NUMERIC",
            "default": "99999",
            "required": True,
        }
    field_def = {
        "name": "SERVICE_CODE",
        "type": "ALPHA_NUMERIC",
        "derived_fields": {"procedure_code": derived_field},
    }
    line = LineWithMeta(line="")
    if record_backed:
        return _create_fields(field_def, "", transformed_value, line)

    visitor = MatchSectionExtractionProcessingVisitor(enabled=True)
    return visitor.create_fields(field_def, "", transformed_value, line)


@pytest.mark.parametrize("record_backed", [False, True])
def test_derived_field_uses_configured_default(record_backed: bool) -> None:
    fields = _build_fields(record_backed, {"procedure_code": None})

    procedure_code = next(
        field for field in fields if field.field_name == "PROCEDURE_CODE"
    )
    assert procedure_code.value == "99999"
    assert procedure_code.value_original == "99999"
    assert procedure_code.field_type == "ALPHA_NUMERIC"
    assert procedure_code.is_required is True


@pytest.mark.parametrize("record_backed", [False, True])
def test_derived_field_preserves_transformed_value(record_backed: bool) -> None:
    fields = _build_fields(record_backed, {"procedure_code": "A1234"})

    procedure_code = next(
        field for field in fields if field.field_name == "PROCEDURE_CODE"
    )
    assert procedure_code.value == "A1234"
    assert procedure_code.value_original is None


@pytest.mark.parametrize("record_backed", [False, True])
def test_derived_field_preserves_string_mapping(record_backed: bool) -> None:
    fields = _build_fields(
        record_backed,
        {"procedure_code": "A1234"},
        derived_field="PROCEDURE_CODE",
    )

    procedure_code = next(
        field for field in fields if field.field_name == "PROCEDURE_CODE"
    )
    assert procedure_code.value == "A1234"


def test_record_backed_row_adds_required_default_without_source_column() -> None:
    visitor = RecordBackedMatchSectionPopulationVisitor(enabled=True)
    match_section = MatchSection()

    visitor._populate_table(
        context=SimpleNamespace(document=None),
        regions_cfg=[
            {
                "title": "SERVICE LINES",
                "type": "table",
                "table": {
                    "body": {
                        "columns": {
                            "SERVICE_CODE": {"annotation_selectors": ["SERVICE_CODE"]}
                        }
                    }
                },
            }
        ],
        match_section=match_section,
        claim_record={"service_lines": {"columns": ["UNUSED"], "rows": [[""]]}},
        role="service_lines",
        section_title="SERVICE LINES",
        template_fields_repeating={
            "SERVICE_CODE": {
                "type": "ALPHA_NUMERIC",
                "derived_fields": {
                    "procedure_code": {
                        "name": "PROCEDURE_CODE",
                        "default": "99999",
                        "required": True,
                    }
                },
            }
        },
    )

    procedure_code = next(
        field
        for field in match_section.matched_field_rows[0].fields
        if field.field_name == "PROCEDURE_CODE"
    )
    assert procedure_code.value == "99999"
