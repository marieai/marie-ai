# python
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Reuse your existing types and helpers (already implemented in your codebase)
from marie.extract.models.match import Span
from marie.extract.parser.base import (
    normalize_row_values,
    parse_bullets_to_kvlist,
    parse_markdown_table,
    split_markdown_sections,
)
from marie.extract.processor.page_span import PageSpan
from marie.extract.structures.cell_with_meta import CellWithMeta
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.structured_region import (
    KeyValue,
    KVList,
    RowRole,
    Section,
    SectionRole,
    StructuredRegion,
    TableRow,
    TableSeries,
    ValueType,
    build_structured_region,
    build_table_series_from_pagespan,
)


class MarkdownRegionParser:
    """
    Generalized Markdown -> StructuredRegion parser.

    Responsibilities:
    - Split markdown into sections via '## ' headings.
    - Parse specific sections into KV lists or tables based on a configurable mapping.
    - Build a StructuredRegion with a PageSpan footprint.
    - Build a single- or multi-page TableSeries via a configurable split policy.

    Notes:
    - You already have the underlying helpers; this class orchestrates them.
    - Uses your existing TableRow, KVList, Section, TableSeries, and build_* utilities.
    """

    def __init__(
        self,
        section_roles: Optional[Dict[str, SectionRole]] = None,
        table_section_titles: Optional[Iterable[str]] = None,
        kv_section_titles: Optional[Iterable[str]] = None,
        # For multi-page tables
        row_split_policy: Optional[
            Callable[[List[List[str]]], Tuple[List[List[str]], List[List[str]]]]
        ] = None,
        # Controls how header is bound in multi-page segmentation: start/single pages get the header
        include_header_on_start_only: bool = True,
    ):
        """
        - section_roles: map of section title (case-insensitive) -> SectionRole.
                         Titles default to MAIN if unspecified.
        - table_section_titles: which sections to parse as tables (case-insensitive).
        - kv_section_titles: which sections to parse as KV lists (case-insensitive).
        - row_split_policy: function that splits rows into (rows_page_a, rows_page_b) for 2-page demos or custom logic.
                            If None, all rows stay on a single page (single-page parsing).
        - include_header_on_start_only: controls header binding propagation across segments.
        """
        self.section_roles = {k.upper(): v for k, v in (section_roles or {}).items()}
        self.table_section_titles = {t.upper() for t in (table_section_titles or [])}
        self.kv_section_titles = {t.upper() for t in (kv_section_titles or [])}
        self.row_split_policy = row_split_policy
        self.include_header_on_start_only = include_header_on_start_only

    # ---------------------------- Public API ----------------------------

    def build_single_page_region(
        self,
        md: str,
        prototype_table,
        *,
        region_id: str,
        page: int,
        page_y: int,
        page_h: int,
    ) -> StructuredRegion:
        """
        Parse markdown into a single-page StructuredRegion. All table rows live on the given page.
        """
        sections_map = split_markdown_sections(md)

        # Build blocks per section
        sections: List[Section] = []
        for title_uc, body in sections_map.items():
            role = self._role_for(title_uc)

            # KV sections (if filtered, otherwise parse all as KV by default)
            kv_items: Optional[List[KeyValue]] = None
            if not self.kv_section_titles or title_uc in self.kv_section_titles:
                kv_items = parse_bullets_to_kvlist(body)
                if kv_items:
                    sections.append(
                        Section(
                            title=self._as_title(title_uc),
                            role=role,
                            blocks=[KVList(type="kvlist", items=kv_items)],
                        )
                    )

            # Table sections (if filtered, otherwise parse all tables by default)
            headers, rows = parse_markdown_table(body)
            if headers:
                table_rows: List[TableRow] = []
                # Header row (anchored at page_y)
                header_cells = [self._text_cell(h) for h in headers]
                table_rows.append(
                    TableRow(
                        role=RowRole.HEADER,
                        cells=header_cells,
                        source_page=page,
                        source_line_ids=[page_y],
                    )
                )
                # Body rows
                next_line_id = page_y + 1
                for r in rows:
                    vals = normalize_row_values(headers, r)
                    cells = [self._text_cell(v) for v in vals]
                    table_rows.append(
                        TableRow(
                            role=RowRole.BODY,
                            cells=cells,
                            source_page=page,
                            source_line_ids=[next_line_id],
                        )
                    )
                    next_line_id += 1

                # PageSpan for the table series
                series_span = PageSpan()
                series_span.add(
                    Span(
                        page=page, y=page_y, h=page_h, msg=f"{self._as_title(title_uc)}"
                    )
                )

                series = build_table_series_from_pagespan(
                    series_id=f"{title_uc.lower().replace(' ', '_')}:p{page}",
                    pagespan=series_span,
                    table=prototype_table,
                    all_rows=table_rows,
                    header_binding=headers,
                )
                sections.append(
                    Section(title=self._as_title(title_uc), role=role, blocks=[series])
                )

        # Region span (single-page)
        region_span = PageSpan()
        region_span.add(Span(page=page, y=page_y, h=page_h, msg="region"))

        return build_structured_region(
            region_id=region_id, region_span=region_span, sections=sections
        )

    def build_multi_page_region(
        self,
        md: str,
        prototype_table,
        *,
        region_id: str,
        # Page 1 footprint
        p1_page: int,
        p1_y: int,
        p1_h: int,
        # Page 2 footprint
        p2_page: int,
        p2_y: int,
        p2_h: int,
    ) -> StructuredRegion:
        """
        Parse markdown into a two-page StructuredRegion (generic multi-page demo).
        The rows are split via row_split_policy. If no table in a section or no split policy,
        table falls back to single page (p1_page).
        """
        sections_map = split_markdown_sections(md)

        sections: List[Section] = []

        for title_uc, body in sections_map.items():
            role = self._role_for(title_uc)

            # KV sections
            if not self.kv_section_titles or title_uc in self.kv_section_titles:
                kv_items = parse_bullets_to_kvlist(body)
                if kv_items:
                    sections.append(
                        Section(
                            title=self._as_title(title_uc),
                            role=role,
                            blocks=[KVList(type="kvlist", items=kv_items)],
                        )
                    )

            # Table sections
            headers, rows = parse_markdown_table(body)
            if not headers:
                continue

            # If a split policy is provided, split rows across the two pages
            if self.row_split_policy:
                rows_p1, rows_p2 = self.row_split_policy(rows)
            else:
                rows_p1, rows_p2 = rows, []

            table_rows: List[TableRow] = []

            # Header on page 1
            table_rows.append(
                TableRow(
                    role=RowRole.HEADER,
                    cells=[self._text_cell(h) for h in headers],
                    source_page=p1_page,
                    source_line_ids=[p1_y],
                )
            )

            lid = p1_y + 1
            for r in rows_p1:
                vals = normalize_row_values(headers, r)
                table_rows.append(
                    TableRow(
                        role=RowRole.BODY,
                        cells=[self._text_cell(v) for v in vals],
                        source_page=p1_page,
                        source_line_ids=[lid],
                    )
                )
                lid += 1

            lid = p2_y + 1
            for r in rows_p2:
                vals = normalize_row_values(headers, r)
                table_rows.append(
                    TableRow(
                        role=RowRole.BODY,
                        cells=[self._text_cell(v) for v in vals],
                        source_page=p2_page,
                        source_line_ids=[lid],
                    )
                )
                lid += 1

            # Combined region/table span (two pages)
            series_span = PageSpan()
            series_span.add(
                Span(page=p1_page, y=p1_y, h=p1_h, msg=f"{self._as_title(title_uc)}:p1")
            )
            series_span.add(
                Span(page=p2_page, y=p2_y, h=p2_h, msg=f"{self._as_title(title_uc)}:p2")
            )

            series = build_table_series_from_pagespan(
                series_id=f"{title_uc.lower().replace(' ', '_')}:p{p1_page}-{p2_page}",
                pagespan=series_span,
                table=prototype_table,
                all_rows=table_rows,
                header_binding=headers if self.include_header_on_start_only else None,
            )

            sections.append(
                Section(title=self._as_title(title_uc), role=role, blocks=[series])
            )

        # Region covering both pages
        region_span = PageSpan()
        region_span.add(Span(page=p1_page, y=p1_y, h=p1_h, msg="p1"))
        region_span.add(Span(page=p2_page, y=p2_y, h=p2_h, msg="p2"))

        return build_structured_region(
            region_id=region_id, region_span=region_span, sections=sections
        )

    # ---------------------------- Helpers ----------------------------

    def _role_for(self, title_uc: str) -> SectionRole:
        # Use provided mapping; default MAIN
        return self.section_roles.get(title_uc, SectionRole.MAIN)

    def _as_title(self, title_uc: str) -> str:
        # Title-case-ish; adjust if you prefer original case preservation
        return title_uc.title()

    @staticmethod
    def _text_cell(text: str) -> CellWithMeta:
        return CellWithMeta(lines=[LineWithMeta(line=text)])
