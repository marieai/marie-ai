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


@pytest.fixture
def md_single():
    return """\
## CLAIM INFORMATION
- PATIENT NAME: STAN SMITH
- PATIENT ACCOUNT: MP21001876
- PROVIDER NAME: R WALL MD
- EMPLOYEE: STAN SMITH
- DEDUCTIBLE AMOUNT: $0.00

## SERVICE LINES
| DATE_OF_SERVICE | PROCEDURE_DESCRIPTION | BILLED_AMOUNT | DISCOUNT | ALLOWED_AMOUNT | DEDUCTIBLE | COINSURANCE | COPAY | BALANCE_PAYABLE | REMARK_CODE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06/24/25 | *** OFFICE VISIT (99214) | 354.00 | 88.50 | 265.50 | 0.00 | 0.00 | 35.00 | 230.50 |  |

## TOTALS
- TOTAL BENEFITS PAID: 230.50
- AMOUNT PATIENT OWES PROVIDER: 35.00
"""


@pytest.fixture
def md_multi():
    return """\
## CLAIM INFORMATION
- PATIENT NAME: STAN SMITH
- PATIENT ACCOUNT: MP21001876
- PROVIDER NAME: R WALL MD
- EMPLOYEE: STAN SMITH
- DEDUCTIBLE AMOUNT: $0.00

## SERVICE LINES
| DATE_OF_SERVICE | PROCEDURE_DESCRIPTION | BILLED_AMOUNT | DISCOUNT | ALLOWED_AMOUNT | DEDUCTIBLE | COINSURANCE | COPAY | BALANCE_PAYABLE | REMARK_CODE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06/24/25 | *** OFFICE VISIT (99214) | 354.00 | 88.50 | 265.50 | 0.00 | 0.00 | 0.00 | 96.00 |  |
| 06/24/25 | XR CHEST (71046) | 120.00 | 24.00 | 96.00 | 0.00 | 0.00 | 0.00 | 96.00 |  |
| 06/24/25 | LAB PANEL (80050) | 220.00 | 55.00 | 165.00 | 0.00 | 0.00 | 0.00 | 165.00 |  |

## TOTALS
- TOTAL BENEFITS PAID: 491.50
- AMOUNT PATIENT OWES PROVIDER: 35.00
"""


def build_single_page_region_from_markdown(md: str, page=3, page_y=120, page_h=40) -> StructuredRegion:
    parser = MarkdownRegionParser(
        section_roles={
            "CLAIM INFORMATION": SectionRole.CONTEXT_ABOVE,
            "SERVICE LINES": SectionRole.MAIN,
            "TOTALS": SectionRole.CONTEXT_BELOW,
        },
        table_section_titles=["SERVICE LINES"],
        kv_section_titles=["CLAIM INFORMATION", "TOTALS"],
    )

    region = parser.build_single_page_region(
        md=md,
        region_id=f"claim_region:p{page}",
        page=page,
        page_y=page_y,
        page_h=page_h,
    )
    return region


def build_multi_page_region_from_markdown(
        md: str, page3_y=80, page3_h=160, page4_y=0, page4_h=120
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
        table_section_titles=["SERVICE LINES"],
        kv_section_titles=["CLAIM INFORMATION", "TOTALS"],
        row_split_policy=split_policy,
    )
    region = parser.build_multi_page_region(
        md=md,
        region_id="claim_region:p3-4",
        p1_page=3, p1_y=page3_y, p1_h=page3_h,
        p2_page=4, p2_y=page4_y, p2_h=page4_h,
    )
    return region


def test_single_page_region(md_single):
    region = build_single_page_region_from_markdown(md_single, page=3, page_y=120, page_h=40)

    print('region', region)
    # Region span
    assert region.span is not None
    pages = pagespan_pages(region.span)
    assert pages == [3]

    # Sections
    titles = [s.title for s in region.sections_flat()]
    assert "CLAIM INFORMATION" in titles
    assert "SERVICE LINES" in titles
    assert "TOTALS" in titles

    # SERVICE LINES -> TableSeries with a single segment
    series_list = region.table_series()
    assert len(series_list) == 1
    series = series_list[0]
    assert isinstance(series, TableSeries)
    assert [seg.segment_role for seg in series.segments] == ["single"]
    assert series.unified_header == [
        "DATE_OF_SERVICE", "PROCEDURE_DESCRIPTION", "BILLED_AMOUNT", "DISCOUNT",
        "ALLOWED_AMOUNT", "DEDUCTIBLE", "COINSURANCE", "COPAY", "BALANCE_PAYABLE", "REMARK_CODE"
    ]

    # TOTALS KV present
    TOTALS_secs = [s for s in region.sections_flat() if s.title == "TOTALS"]
    assert TOTALS_secs and isinstance(TOTALS_secs[0].blocks[0], KVList)
    TOTALS_items = TOTALS_secs[0].blocks[0].items
    keys = {kv.key for kv in TOTALS_items}
    assert "TOTAL BENEFITS PAID" in keys
    assert "AMOUNT PATIENT OWES PROVIDER" in keys


def test_multi_page_region(md_multi):
    region = build_multi_page_region_from_markdown(
        md_multi, page3_y=80, page3_h=160, page4_y=0, page4_h=120
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

    # TOTALS section present
    titles = [s.title for s in region.sections_flat()]
    assert "TOTALS" in titles


def test_duplicate_sections_raise():
    md_with_dupes = """\
## CLAIM INFORMATION
- PATIENT NAME: STAN SMITH

## TOTALS
- TOTAL BENEFITS PAID: 100.00

## TOTALS
- AMOUNT PATIENT OWES PROVIDER: 25.00
"""
    import pytest
    with pytest.raises(ValueError) as ei:
        _ = build_single_page_region_from_markdown(md_with_dupes, page=1, page_y=10, page_h=20)
    assert "Duplicate section titles not allowed" in str(ei.value)


def test_untitled_sections_raise():
    md_with_untitled = """\
## 
- SOME KEY: SOME VALUE

## TOTALS
- TOTAL BENEFITS PAID: 100.00
"""
    import pytest
    with pytest.raises(ValueError) as ei:
        _ = build_single_page_region_from_markdown(md_with_untitled, page=1, page_y=10, page_h=20)
    assert "Untitled sections are not allowed" in str(ei.value)
