import json

import pytest

from marie.extract.parser.json_region_parser import JsonRegionParser
from marie.extract.structures.structured_region import (
    KVList,
    SectionRole,
    StructuredRegion,
    TableSeries,
)


@pytest.fixture
def json_parser_cfg():
    """Configuration for JsonRegionParser similar to markdown tests."""
    return {
        "sections": [
            {"title": "CLAIM INFORMATION", "role": "context_above", "parse": "kv"},
            {"title": "SERVICE LINES", "role": "main", "parse": "table"},
            {"title": "TOTALS", "role": "context_below", "parse": "kv"},
            {"title": "ClaimTotals", "role": "context_below", "parse": "kv"},
            {"title": "ServiceLines", "role": "main", "parse": "table"},
        ]
    }


@pytest.fixture
def json_single():
    """Single page JSON data with KV and table sections."""
    return {
        "CLAIM INFORMATION": {
            "PATIENT NAME": "STAN SMITH",
            "PATIENT ACCOUNT": "MP21001876",
            "PROVIDER NAME": "R WALL MD",
            "EMPLOYEE": "STAN SMITH",
        },
        "SERVICE LINES": {
            "columns": [
                "DATES_OF_SERVICE",
                "PROCEDURE_DESCRIPTION",
                "BILLED_AMOUNT",
                "DISCOUNT",
                "ALLOWED_AMOUNT",
                "DEDUCTIBLE",
                "COINSURANCE",
                "COPAY",
            ],
            "rows": [
                ["12/05/24–12/05/24", "96360 - Hydration Iv", "699.00", "0.00", "699.00", "0.00", "0.00", "35.00"],
                ["12/05/24–12/05/24", "36415 - Surgical Serv", "16.00", "0.00", "16.00", "0.00", "0.00", "0.00"],
                ["12/05/24–12/05/24", "80307 - Lab Service", "214.50", "0.00", "214.50", "0.00", "0.00", "195.50"],
            ]
        },
        "TOTALS": {
            "TOTAL BENEFITS PAID": "230.50",
            "AMOUNT PATIENT OWES PROVIDER": "35.00"
        }
    }


@pytest.fixture
def json_multi():
    """Multi-page JSON data for testing page splitting."""
    return {
        "CLAIM INFORMATION": {
            "PATIENT NAME": "JANE DOE",
            "PROVIDER NAME": "MEDICAL CENTER"
        },
        "SERVICE LINES": {
            "columns": ["DATE", "SERVICE", "AMOUNT"],
            "rows": [
                ["01/01/24", "Service A", "100.00"],
                ["01/02/24", "Service B", "200.00"],
                ["01/03/24", "Service C", "300.00"],
                ["01/04/24", "Service D", "400.00"],
            ]
        },
        "TOTALS": {
            "TOTAL": "1000.00"
        }
    }


def build_single_page_region_from_json(
        json_data, page=3, page_y=120, page_h=40, *, cfg: dict = None
) -> StructuredRegion:
    """Build a single-page region from JSON data."""
    parser = JsonRegionParser.from_config(cfg or {})

    region = parser.build_single_page_region(
        json_data=json_data,
        region_id=f"claim_region:p{page}",
        page=page,
        page_y=page_y,
        page_h=page_h,
    )
    return region


def build_multi_page_region_from_json(
        json_data, *, cfg: dict = None
) -> StructuredRegion:
    """Build a multi-page region from JSON data."""
    parser = JsonRegionParser.from_config(cfg or {})

    def split_rows_demo(rows):
        mid = len(rows) // 2
        return rows[:mid], rows[mid:]

    parser.row_split_policy = split_rows_demo

    region = parser.build_multi_page_region(
        json_data=json_data,
        region_id="multi_page_region",
        p1_page=0,
        p1_y=10,
        p1_h=200,
        p2_page=1,
        p2_y=10,
        p2_h=150,
    )
    return region


def test_single_page_region(json_single, json_parser_cfg):
    """Test basic single-page region parsing from JSON."""
    region = build_single_page_region_from_json(
        json_single, page=3, page_y=120, page_h=40, cfg=json_parser_cfg
    )

    assert region.region_id == "claim_region:p3"
    assert region.span is not None
    assert len(region.span.spanned_pages) == 1

    page_span = region.span.spanned_pages[0]
    assert page_span.page == 3
    assert page_span.y == 120
    assert page_span.h == 40

    # Check sections
    sections = region.sections_flat()
    assert len(sections) == 3

    section_titles = [s.title for s in sections]
    assert "CLAIM INFORMATION" in section_titles
    assert "SERVICE LINES" in section_titles
    assert "TOTALS" in section_titles

    # Check CLAIM INFORMATION section (KV)
    claim_sec = region.find_section("CLAIM INFORMATION")
    assert claim_sec is not None
    assert claim_sec.role == SectionRole.CONTEXT_ABOVE  # Use actual enum values
    assert len(claim_sec.blocks) == 1

    kv_block = claim_sec.blocks[0]
    assert isinstance(kv_block, KVList)
    assert len(kv_block.items) == 4

    kv_dict = {item.key: item.value for item in kv_block.items}
    assert kv_dict["PATIENT NAME"] == "STAN SMITH"
    assert kv_dict["PATIENT ACCOUNT"] == "MP21001876"
    assert kv_dict["PROVIDER NAME"] == "R WALL MD"
    assert kv_dict["EMPLOYEE"] == "STAN SMITH"

    # Check SERVICE LINES section (Table)
    service_sec = region.find_section("SERVICE LINES")
    assert service_sec is not None
    assert service_sec.role == SectionRole.MAIN
    assert len(service_sec.blocks) == 1

    table_series = service_sec.blocks[0]
    assert isinstance(table_series, TableSeries)
    assert len(table_series.segments) == 1

    table_block = table_series.segments[0]
    assert table_block.header_binding is not None
    assert len(table_block.header_binding) == 8
    assert table_block.header_binding[0] == "DATES_OF_SERVICE"
    assert table_block.header_binding[1] == "PROCEDURE_DESCRIPTION"

    # Check table rows
    assert len(table_block.rows) == 4  # 1 header + 3 body rows
    header_row = table_block.rows[0]
    assert header_row.role.name == "HEADER"

    body_rows = table_block.rows[1:]
    assert len(body_rows) == 3
    for row in body_rows:
        assert row.role.name == "BODY"
        assert len(row.cells) == 8

    # Check first data row
    first_row = body_rows[0]
    first_row_values = [cell.lines[0].line for cell in first_row.cells]
    assert first_row_values[0] == "12/05/24–12/05/24"
    assert first_row_values[1] == "96360 - Hydration Iv"
    assert first_row_values[2] == "699.00"

    # Check TOTALS section (KV)
    totals_sec = region.find_section("TOTALS")
    assert totals_sec is not None
    assert totals_sec.role == SectionRole.CONTEXT_BELOW
    assert len(totals_sec.blocks) == 1

    totals_kv = totals_sec.blocks[0]
    assert isinstance(totals_kv, KVList)
    totals_dict = {item.key: item.value for item in totals_kv.items}
    assert totals_dict["TOTAL BENEFITS PAID"] == "230.50"
    assert totals_dict["AMOUNT PATIENT OWES PROVIDER"] == "35.00"


def test_multi_page_region(json_multi, json_parser_cfg):
    """Test multi-page region parsing from JSON."""
    region = build_multi_page_region_from_json(json_multi, cfg=json_parser_cfg)

    assert region.region_id == "multi_page_region"
    assert region.span is not None
    assert len(region.span.spanned_pages) == 2

    # Check page spans
    page1_span = region.span.spanned_pages[0]
    page2_span = region.span.spanned_pages[1]
    assert page1_span.page == 0
    assert page2_span.page == 1

    # Check sections exist
    sections = region.sections_flat()
    section_titles = [s.title for s in sections]
    assert "CLAIM INFORMATION" in section_titles
    assert "SERVICE LINES" in section_titles
    assert "TOTALS" in section_titles

    # Check that table is split across pages
    service_sec = region.find_section("SERVICE LINES")
    table_series = service_sec.blocks[0]

    # Should have segments for both pages
    assert len(table_series.segments) >= 1

    # Check that rows are distributed
    total_body_rows = sum(
        len([r for r in segment.rows if r.role.name == "BODY"])
        for segment in table_series.segments
    )
    assert total_body_rows == 4  # All 4 data rows should be present


def test_json_string_input(json_single, json_parser_cfg):
    """Test that parser accepts JSON string input."""
    json_string = json.dumps(json_single)
    region = build_single_page_region_from_json(
        json_string, page=1, page_y=0, page_h=100, cfg=json_parser_cfg
    )

    assert region.region_id == "claim_region:p1"
    sections = region.sections_flat()
    assert len(sections) == 3


def test_json_array_of_objects_table():
    """Test JSON array of objects table format."""
    json_data = {
        "ServiceLines": [
            {"DATE": "01/01/24", "SERVICE": "Test A", "AMOUNT": "100.00"},
            {"DATE": "01/02/24", "SERVICE": "Test B", "AMOUNT": "200.00"},
        ]
    }

    cfg = {
        "sections": [
            {"title": "ServiceLines", "role": "main", "parse": "table"}
        ]
    }

    region = build_single_page_region_from_json(json_data, cfg=cfg)

    service_sec = region.find_section("SERVICELINES")
    assert service_sec is not None

    table_series = service_sec.blocks[0]
    table_block = table_series.segments[0]

    # Check headers
    assert table_block.header_binding == ["DATE", "SERVICE", "AMOUNT"]

    # Check data rows
    body_rows = [r for r in table_block.rows if r.role.name == "BODY"]
    assert len(body_rows) == 2

    first_row_values = [cell.lines[0].line for cell in body_rows[0].cells]
    assert first_row_values == ["01/01/24", "Test A", "100.00"]


def test_json_single_object_table():
    """Test JSON single object treated as single-row table."""
    json_data = {
        "ClaimTotals": {
            "TOTAL_PAID": "500.00",
            "BALANCE_DUE": "50.00"
        }
    }

    cfg = {
        "sections": [
            {"title": "ClaimTotals", "role": "context_below", "parse": "table"}
        ]
    }

    region = build_single_page_region_from_json(json_data, cfg=cfg)

    totals_sec = region.find_section("CLAIMTOTALS")
    assert totals_sec is not None

    table_series = totals_sec.blocks[0]
    table_block = table_series.segments[0]

    # Check headers (keys from object)
    assert set(table_block.header_binding) == {"TOTAL_PAID", "BALANCE_DUE"}

    # Check single data row
    body_rows = [r for r in table_block.rows if r.role.name == "BODY"]
    assert len(body_rows) == 1


def test_invalid_json_string():
    """Test that invalid JSON string raises appropriate error."""
    cfg = {
        "sections": [
            {"title": "Test", "role": "main", "parse": "kv"}
        ]
    }

    parser = JsonRegionParser.from_config(cfg)

    with pytest.raises(ValueError, match="Invalid JSON format"):
        parser.build_single_page_region(
            json_data='{"invalid": json}',  # Invalid JSON
            region_id="test",
            page=0,
            page_y=0,
            page_h=100,
        )


def test_invalid_json_type():
    """Test that non-dict/string input raises appropriate error."""
    cfg = {
        "sections": [
            {"title": "Test", "role": "main", "parse": "kv"}
        ]
    }

    parser = JsonRegionParser.from_config(cfg)

    with pytest.raises(ValueError, match="json_data must be a JSON string or dictionary"):
        parser.build_single_page_region(
            json_data=["not", "a", "dict"],  # Invalid type
            region_id="test",
            page=0,
            page_y=0,
            page_h=100,
        )


def test_non_object_json_root():
    """Test that JSON array at root raises appropriate error."""
    cfg = {
        "sections": [
            {"title": "Test", "role": "main", "parse": "kv"}
        ]
    }

    parser = JsonRegionParser.from_config(cfg)

    with pytest.raises(ValueError, match="JSON root must be an object/dictionary"):
        parser.build_single_page_region(
            json_data='["array", "at", "root"]',  # Array instead of object
            region_id="test",
            page=0,
            page_y=0,
            page_h=100,
        )


def test_empty_sections():
    """Test that empty JSON object produces no sections."""
    cfg = {
        "sections": [
            {"title": "Test", "role": "main", "parse": "kv"}
        ]
    }

    region = build_single_page_region_from_json({}, cfg=cfg)
    sections = region.sections_flat()
    assert len(sections) == 0


def test_kv_section_filtering():
    """Test that KV section filtering works correctly."""
    json_data = {
        "ALLOWED_SECTION": {
            "KEY1": "VALUE1",
            "KEY2": "VALUE2"
        },
        "FILTERED_SECTION": {
            "KEY3": "VALUE3"
        }
    }

    cfg = {
        "sections": [
            {"title": "ALLOWED_SECTION", "role": "main", "parse": "kv"},
            {"title": "FILTERED_SECTION", "role": "main", "parse": "table"},  # Only table, not kv
        ]
    }

    region = build_single_page_region_from_json(json_data, cfg=cfg)
    sections = region.sections_flat()

    # Should have both sections, but FILTERED_SECTION as table, not KV
    assert len(sections) == 2

    # Check ALLOWED_SECTION has KV
    allowed_sec = region.find_section("ALLOWED_SECTION")
    assert allowed_sec is not None
    assert isinstance(allowed_sec.blocks[0], KVList)

    # Check FILTERED_SECTION has table (single object converted to table)
    filtered_sec = region.find_section("FILTERED_SECTION")
    assert filtered_sec is not None
    assert isinstance(filtered_sec.blocks[0], TableSeries)


def test_table_section_filtering():
    """Test that table section filtering works correctly."""
    json_data = {
        "TABLE_SECTION": {
            "columns": ["A", "B"],
            "rows": [["1", "2"]]
        },
        "KV_SECTION": {
            "columns": ["C", "D"],
            "rows": [["3", "4"]]
        }
    }

    cfg = {
        "sections": [
            {"title": "TABLE_SECTION", "role": "main", "parse": "table"},
            {"title": "KV_SECTION", "role": "main", "parse": "kv"},  # Only kv, not table
        ]
    }

    region = build_single_page_region_from_json(json_data, cfg=cfg)
    sections = region.sections_flat()

    # Should have both sections
    assert len(sections) == 2

    # Only TABLE_SECTION should have table block
    table_sections = [s for s in sections if any(isinstance(b, TableSeries) for b in s.blocks)]
    assert len(table_sections) == 1
    assert table_sections[0].title == "TABLE_SECTION"

    # KV_SECTION should have KV block (columns/rows converted to key-value pairs)
    kv_sections = [s for s in sections if any(isinstance(b, KVList) for b in s.blocks)]
    assert len(kv_sections) == 1
    assert kv_sections[0].title == "KV_SECTION"


def test_complex_kv_values():
    """Test that complex JSON values are properly serialized in KV sections."""
    json_data = {
        "COMPLEX_DATA": {
            "simple_string": "test",
            "number": 42,
            "boolean": True,
            "null_value": None,
            "nested_object": {"inner": "value"},
            "array": [1, 2, 3]
        }
    }

    cfg = {
        "sections": [
            {"title": "COMPLEX_DATA", "role": "main", "parse": "kv"}
        ]
    }

    region = build_single_page_region_from_json(json_data, cfg=cfg)
    sections = region.sections_flat()
    assert len(sections) == 1

    kv_block = sections[0].blocks[0]
    kv_dict = {item.key: item.value for item in kv_block.items}

    # Check various value types
    assert kv_dict["simple_string"] == "test"
    assert kv_dict["number"] == "42"
    assert kv_dict["boolean"] == "True"
    assert kv_dict["null_value"] == "None"
    assert kv_dict["nested_object"] == '{"inner": "value"}'
    assert kv_dict["array"] == "[1, 2, 3]"