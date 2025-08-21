from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
from marie.extract.structures.table import Table
from marie.extract.structures.table_metadata import TableMetadata


class MarkdownRegionParser:
    """
    Generalized Markdown -> StructuredRegion parser.

    Responsibilities:
    - Split markdown into sections via '## ' headings.
    - Parse specific sections into KV lists or tables based on a configurable mapping.
    - Build a StructuredRegion with a PageSpan footprint.
    - Build a single- or multi-page TableSeries via a configurable split policy.

    Notes:
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
        *,
        region_id: str,
        page: int,
        page_y: int,
        page_h: int,
    ) -> StructuredRegion:
        """
        Parse markdown into a single-page StructuredRegion. All table rows live on the given page.
        """
        # Use a duplicate-preserving section parser and validate titles
        sections_seq = self._iter_sections(md)
        self._validate_sections(sections_seq)

        # Build blocks per section
        sections: List[Section] = []
        for title_uc, body in sections_seq:
            role = self._role_for(title_uc)

            # KV section (optional)
            kv_sec = self._build_kv_section(title_uc, role, body)
            if kv_sec is not None:
                sections.append(kv_sec)

            # Table section (optional)
            headers, rows = parse_markdown_table(body)
            if headers:
                # Plan a single segment on 'page'
                segments = [
                    {"page": page, "y": page_y, "h": page_h, "rows": rows},
                ]
                tbl_sec = self._build_table_section(
                    title_uc=title_uc,
                    role=role,
                    headers=headers,
                    row_segments=segments,
                    header_on_first_only=True,
                )
                if tbl_sec is not None:
                    sections.append(tbl_sec)

        # Region span (single-page)
        region_span = PageSpan()
        region_span.add(Span(page=page, y=page_y, h=page_h, msg="region"))

        return build_structured_region(
            region_id=region_id, region_span=region_span, sections=sections
        )

    def build_multi_page_region(
        self,
        md: str,
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
        # Use a duplicate-preserving section parser and validate titles
        sections_seq = self._iter_sections(md)
        self._validate_sections(sections_seq)

        sections: List[Section] = []

        for title_uc, body in sections_seq:
            role = self._role_for(title_uc)

            # KV section (optional)
            kv_sec = self._build_kv_section(title_uc, role, body)
            if kv_sec is not None:
                sections.append(kv_sec)

            # Table sections
            headers, rows = parse_markdown_table(body)
            if not headers:
                continue

            # Split rows across two pages (or keep all on first if no policy)
            if self.row_split_policy:
                rows_p1, rows_p2 = self.row_split_policy(rows)
            else:
                rows_p1, rows_p2 = rows, []

            segments = [
                {"page": p1_page, "y": p1_y, "h": p1_h, "rows": rows_p1},
                {"page": p2_page, "y": p2_y, "h": p2_h, "rows": rows_p2},
            ]

            tbl_sec = self._build_table_section(
                title_uc=title_uc,
                role=role,
                headers=headers,
                row_segments=segments,
                header_on_first_only=self.include_header_on_start_only,
            )
            if tbl_sec is not None:
                sections.append(tbl_sec)

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
        # Preserve provided uppercase title
        return title_uc

    @staticmethod
    def _text_cell(text: str) -> CellWithMeta:
        return CellWithMeta(lines=[LineWithMeta(line=text)])

    def _iter_sections(self, md: str) -> List[Tuple[str, str]]:
        """
        Parse markdown into an ordered list of (TITLE_UC, BODY) pairs,
        preserving duplicates and empty titles for validation upstream.
        """
        import re

        text = md.strip()
        if not text:
            return []
        headings = list(re.finditer(r"(?m)^##\s*(.*)$", text))
        sections: List[Tuple[str, str]] = []
        if not headings:
            return sections
        for idx, m in enumerate(headings):
            title_raw = (m.group(1) or "").strip()
            start = m.end()
            end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
            body = text[start:end].strip()
            sections.append((title_raw.upper(), body))
        return sections

    def _validate_sections(self, sections: List[Tuple[str, str]]) -> None:
        """
        - No untitled sections (empty or whitespace titles)
        - No duplicate section titles (case-insensitive)
        """
        untitled = [t for t, _ in sections if not t.strip()]
        if untitled:
            raise ValueError("Untitled sections are not allowed")

        seen: set[str] = set()
        dups: List[str] = []
        for t, _ in sections:
            if t in seen and t not in dups:
                dups.append(t)
            seen.add(t)
        if dups:
            raise ValueError(f"Duplicate section titles not allowed: {', '.join(dups)}")

    def _build_kv_section(
        self, title_uc: str, role: SectionRole, body: str
    ) -> Optional[Section]:
        """
        Build a KVList Section if bullet kvs exist and allowed by kv_section_titles.
        """
        if self.kv_section_titles and title_uc not in self.kv_section_titles:
            return None
        kv_items = parse_bullets_to_kvlist(body)
        if not kv_items:
            return None
        return Section(
            title=self._as_title(title_uc),
            role=role,
            blocks=[KVList(type="kvlist", items=kv_items)],
        )

    def _build_table_section(
        self,
        *,
        title_uc: str,
        role: SectionRole,
        headers: List[str],
        row_segments: List[Dict[str, object]],
        header_on_first_only: bool,
    ) -> Optional[Section]:
        """
        DRY builder for a table-backed Section.
        row_segments is a list of dicts: {"page": int, "y": int, "h": int, "rows": List[List[str]]}
        """
        # Reject if table section filtering is set and this section is not included
        if self.table_section_titles and title_uc not in self.table_section_titles:
            return None

        # Build TableRows for segments, header only on first segment
        table_rows: List[TableRow] = []

        # Header cells (used for both Table grid and optional header row)
        header_cells = [self._text_cell(h) for h in headers]

        # Build series PageSpan across all segments
        series_span = PageSpan()

        # Accumulate all body cells (for Table grid)
        body_cells_grid: List[List[CellWithMeta]] = []

        for seg_idx, seg in enumerate(row_segments):
            seg_page = int(seg["page"])
            seg_y = int(seg["y"])
            seg_h = int(seg["h"])
            seg_rows: List[List[str]] = list(seg["rows"])  # type: ignore

            # Skip empty segments without contributing span/rows
            if (
                seg_h <= 0
                and not seg_rows
                and not (seg_idx == 0 and header_on_first_only)
            ):
                continue

            # Add span for this segment
            series_span.add(
                Span(page=seg_page, y=seg_y, h=seg_h, msg=f"{self._as_title(title_uc)}")
            )

            # Optional header row only on the first segment
            if seg_idx == 0:
                table_rows.append(
                    TableRow(
                        role=RowRole.HEADER,
                        cells=header_cells,
                        source_page=seg_page,
                        source_line_ids=[seg_y],
                    )
                )

            # Body rows for this segment
            lid = seg_y + 1
            for r in seg_rows:
                vals = normalize_row_values(headers, r)
                cells = [self._text_cell(v) for v in vals]
                body_cells_grid.append(cells)
                table_rows.append(
                    TableRow(
                        role=RowRole.BODY,
                        cells=cells,
                        source_page=seg_page,
                        source_line_ids=[lid],
                    )
                )
                lid += 1

        # If no rows at all, do not emit a table section
        if len(table_rows) == 0:
            return None

        # Real Table construction (header + body)
        table_cells: List[List[CellWithMeta]] = [header_cells] + body_cells_grid
        # Use the first segment page/y for metadata defaults
        first = row_segments[0]
        table_metadata = TableMetadata(
            page_id=int(first["page"]), title="extracted", line=int(first["y"])
        )
        table_obj = Table(cells=table_cells, metadata=table_metadata)

        # Build TableSeries (header_binding for start/single pages only)
        series = build_table_series_from_pagespan(
            series_id=f"{title_uc.lower().replace(' ', '_')}:p{','.join(str(int(seg['page'])) for seg in row_segments)}",
            pagespan=series_span,
            table=table_obj,
            all_rows=table_rows,
            header_binding=headers if header_on_first_only else None,
        )

        return Section(
            title=self._as_title(title_uc),
            role=role,
            blocks=[series],
        )
