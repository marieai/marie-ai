from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, Iterable, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from marie.extract.models.match import Span
from marie.extract.processor.page_span import PageSpan
from marie.extract.structures.cell_with_meta import CellWithMeta
from marie.extract.structures.table import Table

# ===================================================================
#                           PRIMITIVES
# ===================================================================


class ValueType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    MONEY = "money"
    DATE = "date"
    CODE = "code"
    NAME = "name"
    ID = "id"
    UNKNOWN = "unknown"


class TextSpan(BaseModel):
    text: str
    span: Optional[Span] = None
    lang: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("span", when_used="json")
    def _ser_span(self, span: Optional[Span], _info):
        if not span:
            return None
        return {
            "page": span.page,
            "y": span.y,
            "h": span.h,
            "msg": getattr(span, "msg", None),
        }


class KeyValue(BaseModel):
    key: str
    value: str
    value_type: ValueType = ValueType.UNKNOWN
    normalized: Optional[str] = None
    source: Optional[TextSpan] = None
    tags: Dict[str, str] = Field(default_factory=dict)


class RowRole(str, Enum):
    HEADER = "header"
    SUBHEADER = "subheader"
    BODY = "body"
    FOOTER = "footer"
    TOTALS = "totals"
    SPACER = "spacer"


class TableRow(BaseModel):
    """
    A logical row extracted from OCR/layout. Keep source_line_ids page-scoped so
    we can slice by Span(y, h) which are based on line_ids in your PageSpan.create().
    """

    role: RowRole = RowRole.BODY
    cells: List["CellWithMeta"]  # reuse your class
    source_page: Optional[int] = None  # (optional) if every row is page-local
    source_line_ids: Optional[List[int]] = None  # line_id(s) for intersection with Span
    tags: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ===================================================================
#                            BLOCKS
# ===================================================================


class BlockBase(BaseModel):
    """
    Abstract base for any block. The 'span' is a PageSpan (multi-page footprint).
    """

    block_id: Optional[str] = None
    span: Optional[PageSpan] = None
    tags: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("span", when_used="json")
    def _ser_pagespan(self, span: Optional[PageSpan], _info):
        if not span:
            return None
        return [
            {"page": s.page, "y": s.y, "h": s.h, "msg": getattr(s, "msg", None)}
            for s in span.spanned_pages
        ]


class KVList(BlockBase):
    type: Literal["kvlist"] = "kvlist"
    items: List[KeyValue] = Field(default_factory=list)


class FreeText(BlockBase):
    type: Literal["text"] = "text"
    runs: List[TextSpan] = Field(default_factory=list)


class TableBlock(BlockBase):
    """
    A single-page (or page-slice) of a table. Keep your 'Table' grid as canonical.
    """

    type: Literal["table"] = "table"
    table: "Table"  # your existing grid object
    rows: List[TableRow] = Field(default_factory=list)
    header_binding: Optional[List[str]] = None  # collapsed final header per column
    segment_role: Literal["start", "middle", "end", "single"] = "single"
    continuation_hint: Optional[str] = None  # e.g., "continued on next page"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TableSeries(BlockBase):
    """
    A multi-page table composed of ordered TableBlock segments.
    """

    type: Literal["table_series"] = "table_series"
    series_id: Optional[str] = None
    segments: List[TableBlock] = Field(default_factory=list)
    unified_header: Optional[List[str]] = None  # consumer-facing header

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ---- Helpers ----
    def pages(self) -> List[int]:
        pages: List[int] = []
        for seg in self.segments:
            if seg.span:
                pages.extend([s.page for s in seg.span.spanned_pages])
        return sorted(set(pages))

    def iter_rows(self, roles: Optional[List[RowRole]] = None) -> Iterable[TableRow]:
        def seg_key(seg: TableBlock):
            if seg.span and seg.span.spanned_pages:
                return min(s.page for s in seg.span.spanned_pages)
            return 0

        for seg in sorted(self.segments, key=seg_key):
            for r in seg.rows:
                if (roles is None) or (r.role in roles):
                    yield r


Block = Union[KVList, FreeText, TableBlock, TableSeries]


# ===================================================================
#                           SECTIONS & REGION
# ===================================================================


class SectionRole(str, Enum):
    CONTEXT_ABOVE = "context_above"
    MAIN = "main"
    CONTEXT_BELOW = "context_below"
    SIDEBAR = "sidebar"
    UNKNOWN = "unknown"


class Section(BaseModel):
    title: Optional[str] = None
    role: SectionRole = SectionRole.UNKNOWN
    blocks: List[Block] = Field(default_factory=list)
    span: Optional[PageSpan] = None  # optional: footprint across pages
    tags: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("span", when_used="json")
    def _ser_pagespan(self, span: Optional[PageSpan], _info):
        if not span:
            return None
        return [
            {"page": s.page, "y": s.y, "h": s.h, "msg": getattr(s, "msg", None)}
            for s in span.spanned_pages
        ]


class RegionPart(BaseModel):
    """
    The portion of the region that lives on a *single* page.
    """

    span: Span  # single-page footprint
    sections: List[Section] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("span", when_used="json")
    def _ser_span(self, span: Span, _info):
        return {
            "page": span.page,
            "y": span.y,
            "h": span.h,
            "msg": getattr(span, "msg", None),
        }


class StructuredRegion(BaseModel):
    """
    A logical region that can span multiple pages (e.g., a claim bundle:
    claim KVs + service-line table across pages + totals).
    """

    region_id: Optional[str] = None
    span: Optional[PageSpan] = None
    parts: List[RegionPart] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("span", when_used="json")
    def _ser_pagespan(self, span: Optional[PageSpan], _info):
        if not span:
            return None
        return [
            {"page": s.page, "y": s.y, "h": s.h, "msg": getattr(s, "msg", None)}
            for s in span.spanned_pages
        ]

    # ---- Convenience helpers ----
    def sections_flat(self) -> List[Section]:
        ordered = sorted(self.parts, key=lambda p: p.span.page)
        out: List[Section] = []
        for p in ordered:
            out.extend(p.sections)
        return out

    def table_series(self) -> List[TableSeries]:
        out: List[TableSeries] = []
        for sec in self.sections_flat():
            for b in sec.blocks:
                if isinstance(b, TableSeries):
                    out.append(b)
                elif isinstance(b, TableBlock):
                    out.append(
                        TableSeries(
                            series_id=b.block_id,
                            segments=[b],
                            unified_header=b.header_binding,
                            span=b.span,
                        )
                    )
        return out


# ===================================================================
#                    SPAN / PAGESPAN DECOMPOSITION UTILS
# ===================================================================


def normalize_pagespan(pagespan: PageSpan) -> List[Span]:
    """
    Return the per-page spans (sorted by page). This is a simple decomposition
    of the multi-page PageSpan into single-page Span elements.
    """
    return sorted(list(pagespan.spanned_pages), key=lambda s: s.page)


def pagespan_pages(pagespan: PageSpan) -> List[int]:
    """List of pages covered by the PageSpan."""
    return [s.page for s in normalize_pagespan(pagespan)]


def span_contains_line_id(s: Span, line_id: int) -> bool:
    """
    True if the line_id falls within [y, y+h) of this Span.
    Your PageSpan.create() sets Span.y to the start line_id, and h to count.
    """
    return s.y <= line_id < (s.y + s.h)


def row_intersects_span(row: TableRow, s: Span) -> bool:
    """
    Intersection predicate between a row and a single-page span.
    Uses row.source_line_ids (page-scoped) when available.
    If your rows carry bounding boxes instead, replace this with a bbox check.
    """
    if row.source_line_ids is None:
        return False
    return any(span_contains_line_id(s, lid) for lid in row.source_line_ids)


def slice_rows_by_pagespan(
    rows: List[TableRow],
    pagespan: PageSpan,
    intersects: Optional[Callable[[TableRow, Span], bool]] = None,
) -> Dict[int, List[TableRow]]:
    """
    Decompose a set of rows by the pages covered in 'pagespan'.
    Returns: { page_number: [rows_on_that_page] }
    """
    pred = intersects or row_intersects_span
    buckets: Dict[int, List[TableRow]] = {}
    for s in normalize_pagespan(pagespan):
        per_page = [r for r in rows if pred(r, s)]
        buckets[s.page] = per_page
    return buckets


def build_table_series_from_pagespan(
    series_id: str,
    pagespan: PageSpan,
    table: "Table",
    all_rows: List[TableRow],
    header_binding: Optional[List[str]] = None,
) -> TableSeries:
    """
    Split a table across the given PageSpan into TableBlock segments and
    tag each as start/middle/end/single.
    """
    spans = normalize_pagespan(pagespan)
    page_to_rows = slice_rows_by_pagespan(all_rows, pagespan)

    segments: List[TableBlock] = []
    for idx, s in enumerate(spans):
        if len(spans) == 1:
            role: Literal["start", "middle", "end", "single"] = "single"
        elif idx == 0:
            role = "start"
        elif idx == len(spans) - 1:
            role = "end"
        else:
            role = "middle"

        # Construct a single-page PageSpan for this segment
        seg_span = PageSpan()
        seg_span.add(Span(page=s.page, y=s.y, h=s.h, msg=getattr(s, "msg", "")))

        seg_rows = page_to_rows.get(s.page, [])

        segments.append(
            TableBlock(
                block_id=f"{series_id}:{s.page}",
                type="table",
                span=seg_span,
                table=table,
                rows=seg_rows,
                header_binding=header_binding if role in ("start", "single") else None,
                segment_role=role,
                continuation_hint=(
                    "continued on next page"
                    if role in ("start", "middle") and idx < len(spans) - 1
                    else None
                ),
            )
        )

    return TableSeries(
        type="table_series",
        series_id=series_id,
        span=pagespan,
        segments=segments,
        unified_header=header_binding,
    )


def distribute_sections_into_parts(
    region_span: PageSpan, sections: List[Section]
) -> List[RegionPart]:
    """
    Create RegionPart objects for each page in region_span and place sections into the
    first page by default, unless a section has its own span, in which case we attach
    it to all pages where it overlaps.
    """
    parts: List[RegionPart] = []
    for s in normalize_pagespan(region_span):
        parts.append(
            RegionPart(
                span=Span(page=s.page, y=s.y, h=s.h, msg=getattr(s, "msg", None)),
                sections=[],
            )
        )

    # Helper: does a section overlap a page?
    def section_on_page(sec: Section, page: int) -> bool:
        if not sec.span:
            return False
        return any(sp.page == page for sp in sec.span.spanned_pages)

    if not sections:
        return parts

    # If no section carries its own span, attach all to the first part.
    any_spanned = any(sec.span is not None for sec in sections)
    if not any_spanned:
        if parts:
            parts[0].sections.extend(sections)
        return parts

    # Otherwise, distribute by overlap
    for part in parts:
        page = part.span.page
        for sec in sections:
            if sec.span and section_on_page(sec, page):
                part.sections.append(sec)

    return parts


def build_structured_region(
    region_id: str, region_span: PageSpan, sections: List[Section]
) -> StructuredRegion:
    """
    Build a multi-page StructuredRegion out of a PageSpan and a set of Sections.
    Sections with their own span are distributed to overlapping pages. Others fall back to the first page.
    """
    parts = distribute_sections_into_parts(region_span, sections)
    return StructuredRegion(region_id=region_id, span=region_span, parts=parts)
