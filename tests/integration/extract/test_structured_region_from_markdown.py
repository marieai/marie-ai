# pytest -q tests/test_structured_region_from_markdown.py
import pytest

from marie.extract.parser.markdown_region_parser import MarkdownRegionParser
from marie.extract.structures.structured_region import (
    KVList,
    RowRole,
    SectionRole,
    StructuredRegion,
    TableSeries,
    pagespan_pages,
    slice_rows_by_pagespan,
)


# Tiny stand-in for a Table; tests don’t call any methods on it.
class _FakeTable:
    def __init__(self, table_id="t1", page=None):
        self.metadata = type("M", (), {"table_id": table_id, "page": page})()

# ------------------------------------------------------------------------------------
# Fixtures: sample markdown
# ------------------------------------------------------------------------------------

@pytest.fixture
def md_single():
    return """\
## Claim Information
- PATIENT NAME: STAN SMITH
- PATIENT ACCOUNT: MP21001876
- PROVIDER NAME: R WALL MD
- EMPLOYEE: STAN SMITH
- DEDUCTIBLE AMOUNT: $0.00

## Service Lines
| DATE_OF_SERVICE | PROCEDURE_DESCRIPTION | BILLED_AMOUNT | DISCOUNT | ALLOWED_AMOUNT | DEDUCTIBLE | COINSURANCE | COPAY | BALANCE_PAYABLE | REMARK_CODE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06/24/25 | *** OFFICE VISIT (99214) | 354.00 | 88.50 | 265.50 | 0.00 | 0.00 | 35.00 | 230.50 |  |

## Totals
- TOTAL BENEFITS PAID: 230.50
- AMOUNT PATIENT OWES PROVIDER: 35.00
"""

@pytest.fixture
def md_multi():
    return """\
## Claim Information
- PATIENT NAME: STAN SMITH
- PATIENT ACCOUNT: MP21001876
- PROVIDER NAME: R WALL MD
- EMPLOYEE: STAN SMITH
- DEDUCTIBLE AMOUNT: $0.00

## Service Lines
| DATE_OF_SERVICE | PROCEDURE_DESCRIPTION | BILLED_AMOUNT | DISCOUNT | ALLOWED_AMOUNT | DEDUCTIBLE | COINSURANCE | COPAY | BALANCE_PAYABLE | REMARK_CODE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06/24/25 | *** OFFICE VISIT (99214) | 354.00 | 88.50 | 265.50 | 0.00 | 0.00 | 35.00 | 230.50 |  |
| 06/24/25 | XR CHEST (71046) | 120.00 | 24.00 | 96.00 | 0.00 | 0.00 | 0.00 | 96.00 |  |
| 06/24/25 | LAB PANEL (80050) | 220.00 | 55.00 | 165.00 | 0.00 | 0.00 | 0.00 | 165.00 |  |

## Totals
- TOTAL BENEFITS PAID: 491.50
- AMOUNT PATIENT OWES PROVIDER: 35.00
"""

# ------------------------------------------------------------------------------------
# Builders (now using MarkdownRegionParser)
# ------------------------------------------------------------------------------------

def build_single_page_region_from_markdown(md: str, prototype_table, page=3, page_y=120, page_h=40) -> StructuredRegion:
    parser = MarkdownRegionParser(
        section_roles={
            "CLAIM INFORMATION": SectionRole.CONTEXT_ABOVE,
            "SERVICE LINES": SectionRole.MAIN,
            "TOTALS": SectionRole.CONTEXT_BELOW,
        },
        table_section_titles=["Service Lines"],
        kv_section_titles=["Claim Information", "Totals"],
    )
    region = parser.build_single_page_region(
        md=md,
        prototype_table=prototype_table,
        region_id=f"claim_region:p{page}",
        page=page,
        page_y=page_y,
        page_h=page_h,
    )
    return region

def build_multi_page_region_from_markdown(
        md: str, prototype_table, page3_y=80, page3_h=160, page4_y=0, page4_h=120
) -> StructuredRegion:
    def split_policy(rows):
        # Same demo split policy used in tests: roughly half, but at least one on the first page
        split_at = max(1, len(rows) // 2)
        return rows[:split_at], rows[split_at:]

    parser = MarkdownRegionParser(
        section_roles={
            "CLAIM INFORMATION": SectionRole.CONTEXT_ABOVE,
            "SERVICE LINES": SectionRole.MAIN,
            "TOTALS": SectionRole.CONTEXT_BELOW,
        },
        table_section_titles=["Service Lines"],
        kv_section_titles=["Claim Information", "Totals"],
        row_split_policy=split_policy,
    )
    region = parser.build_multi_page_region(
        md=md,
        prototype_table=prototype_table,
        region_id="claim_region:p3-4",
        p1_page=3, p1_y=page3_y, p1_h=page3_h,
        p2_page=4, p2_y=page4_y, p2_h=page4_h,
    )
    return region

# ------------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------------

def test_single_page_region(md_single):
    # Using a tiny fake Table (you can swap in the real Table if desired)
    proto_table = _FakeTable(table_id="tbl_p3", page=3)

    region = build_single_page_region_from_markdown(md_single, proto_table, page=3, page_y=120, page_h=40)

    # Region span
    assert region.span is not None
    pages = pagespan_pages(region.span)
    assert pages == [3]

    # Sections
    titles = [s.title for s in region.sections_flat()]
    assert "Claim Information" in titles
    assert "Service Lines" in titles
    assert "Totals" in titles

    # Service Lines -> TableSeries with a single segment
    series_list = region.table_series()
    assert len(series_list) == 1
    series = series_list[0]
    assert isinstance(series, TableSeries)
    assert [seg.segment_role for seg in series.segments] == ["single"]
    assert series.unified_header == [
        "DATE_OF_SERVICE", "PROCEDURE_DESCRIPTION", "BILLED_AMOUNT", "DISCOUNT",
        "ALLOWED_AMOUNT", "DEDUCTIBLE", "COINSURANCE", "COPAY", "BALANCE_PAYABLE", "REMARK_CODE"
    ]

    # Totals KV present
    totals_secs = [s for s in region.sections_flat() if s.title == "Totals"]
    assert totals_secs and isinstance(totals_secs[0].blocks[0], KVList)
    totals_items = totals_secs[0].blocks[0].items
    keys = {kv.key for kv in totals_items}
    assert "TOTAL BENEFITS PAID" in keys
    assert "AMOUNT PATIENT OWES PROVIDER" in keys

def test_multi_page_region(md_multi):
    proto_table = _FakeTable(table_id="tbl_p3p4", page=None)

    region = build_multi_page_region_from_markdown(
        md_multi, proto_table, page3_y=80, page3_h=160, page4_y=0, page4_h=120
    )

    # Region spans pages 3 and 4
    assert region.span is not None
    assert pagespan_pages(region.span) == [3, 4]

    # TableSeries has 2 segments: start + end
    series_list = region.table_series()
    assert len(series_list) == 1
    series = series_list[0]
    roles = [seg.segment_role for seg in series.segments]
    assert roles == ["start", "end"]

    # Segment pages are ordered and correct
    seg_pages = sorted([seg.span.spanned_pages[0].page for seg in series.segments])
    assert seg_pages == [3, 4]

    # Basic body-row distribution check: 1 row on p3, 2 rows on p4 (based on split policy)
    rows_by_page = slice_rows_by_pagespan(
        rows=[r for r in series.iter_rows()],  # includes header row; we’ll filter
        pagespan=region.span
    )
    # Filter only BODY rows within each segment
    body_counts = {}
    for seg in series.segments:
        p = seg.span.spanned_pages[0].page
        body_counts[p] = sum(1 for r in seg.rows if r.role == RowRole.BODY)
    assert body_counts[3] == 1
    assert body_counts[4] == 2

    # Totals section present
    titles = [s.title for s in region.sections_flat()]
    assert "Totals" in titles