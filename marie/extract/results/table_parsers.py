from typing import List, Optional

from marie.extract.parser.base import parse_markdown_table
from marie.extract.parser.markdown_region_parser import MarkdownRegionParser
from marie.extract.results.span_util import create_page_span_from_lines
from marie.extract.structures import UnstructuredDocument
from marie.extract.structures.cell_with_meta import CellWithMeta
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.structured_region import (
    RegionPart,
    RowRole,
    Section,
    SectionRole,
    StructuredRegion,
    TableBlock,
    TableRow,
    TableSeries,
)
from marie.extract.structures.table import Table
from marie.extract.structures.table_metadata import TableMetadata
from marie.logging_core.predefined import default_logger as logger


def _parse_table_plain(
    doc: UnstructuredDocument,
    md_content: str,
    page_id: int,
    segment_index: int,
    table_segment_meta: dict,
) -> Optional[StructuredRegion]:
    """Parses a markdown file to extract a table and returns a StructuredRegion."""

    headers, body_rows = parse_markdown_table(md_content)
    if not headers:
        return None

    logger.debug(f"parsed_table => {[headers] + body_rows}")

    header_cells = [CellWithMeta(lines=[LineWithMeta(header)]) for header in headers]
    body_rows_with_meta = [
        [CellWithMeta(lines=[LineWithMeta(cell)]) for cell in row_data]
        for row_data in body_rows
    ]
    table_rows_with_meta = [header_cells] + body_rows_with_meta

    start_line = table_segment_meta["start"]
    end_line = table_segment_meta["end"]

    table_metadata = TableMetadata(
        page_id=page_id,
        title="extracted",
        line=start_line,
    )
    table_obj = Table(cells=table_rows_with_meta, metadata=table_metadata)

    header_tablerow = TableRow(role=RowRole.HEADER, cells=table_obj.cells[0])
    body_tablerows = [
        TableRow(role=RowRole.BODY, cells=row) for row in table_obj.cells[1:]
    ]
    structured_rows: List[TableRow] = [header_tablerow] + body_tablerows

    page_span = create_page_span_from_lines(doc, start_line, end_line)
    if not page_span.spanned_pages:
        logger.error(
            f"Could not create page span for segment {segment_index} on page {page_id}"
        )
        return None

    num_pages = len(page_span.spanned_pages)
    if num_pages > 1:
        raise RuntimeError("Multiple pages extracted. Not yet supported")
    span = page_span.spanned_pages[0]

    table_block = TableBlock(
        table=table_obj,
        rows=structured_rows,
        header_binding=headers,
        span=page_span,
        segment_role="single",
    )
    table_series = TableSeries(
        segments=[table_block], unified_header=headers, span=page_span
    )
    section = Section(
        title="extracted",
        role=SectionRole.MAIN,
        blocks=[table_series],
        span=page_span,
    )
    part = RegionPart(sections=[section], span=span)
    region = StructuredRegion(
        region_id=f"doc_region:p{page_id}_{segment_index}", span=page_span, parts=[part]
    )
    return region


def _parse_table_mrp(
    doc: UnstructuredDocument,
    md_content: str,
    page_id: int,
    segment_index: int,
    table_segment_meta: dict,
    cfg: dict,
    **kwargs,
) -> Optional[StructuredRegion]:
    """Parses a markdown file to extract tables using MarkdownRegionParser."""

    if not md_content.strip():
        return None

    parser = MarkdownRegionParser.from_config(cfg)
    logger.info(f"parser => {[parser.summary()]}")

    # Deprecated way to configure the parser
    # section_roles = kwargs.get("section_roles", {})
    # table_section_titles = kwargs.get("table_section_titles", [])
    # kv_section_titles = kwargs.get("kv_section_titles", [])
    #
    # if not table_section_titles:
    #     logger.warning("table_section_titles not provided for 'mrp' parsing method.")
    #     return None
    #
    # parser = MarkdownRegionParser(
    #     section_roles=section_roles,
    #     table_section_titles=table_section_titles,
    #     kv_section_titles=kv_section_titles,
    # )

    # TODO : Implement the multi-page span
    start_line = table_segment_meta["start"]
    end_line = table_segment_meta["end"]

    page_span = create_page_span_from_lines(doc, start_line, end_line)
    if not page_span.spanned_pages:
        logger.error(
            f"Could not create page span for segment {segment_index} on page {page_id}"
        )
        return None

    num_pages = len(page_span.spanned_pages)
    if num_pages > 1:
        raise RuntimeError("Multiple pages extracted. Not yet supported")
    span = page_span.spanned_pages[0]

    region = parser.build_single_page_region(
        md=md_content,
        region_id=f"doc_region:p{page_id}",
        page=page_id,
        page_y=span.y,
        page_h=span.h,
    )

    return region
