from __future__ import annotations

import re
from typing import List, Optional, Tuple

from marie.extract.parser.base import parse_bullets_to_kvlist, parse_markdown_table
from marie.extract.parser.base_region_parser import BaseRegionParser
from marie.extract.structures.structured_region import (
    KVList,
    Section,
    SectionRole,
    StructuredRegion,
)


class MarkdownRegionParser(BaseRegionParser):
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
        return super().build_single_page_region(
            data=md,
            region_id=region_id,
            page=page,
            page_y=page_y,
            page_h=page_h,
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
        """
        return super().build_multi_page_region(
            data=md,
            region_id=region_id,
            p1_page=p1_page,
            p1_y=p1_y,
            p1_h=p1_h,
            p2_page=p2_page,
            p2_y=p2_y,
            p2_h=p2_h,
        )

    def _parse_and_validate_sections(self, md: str) -> List[Tuple[str, str]]:
        """
        Common entry to parse sections and enforce validation rules eagerly:
        - No '##' headings without a title
        - No duplicate titles (case-insensitive)
        """

        # Fast guard: explicit untitled heading lines like '##' or '##   '
        if re.search(r"(?m)^##\s*$", md or ""):
            raise ValueError("Untitled sections are not allowed")

        sections_seq = self._iter_sections(md)
        self._validate_sections(sections_seq)
        return sections_seq

    def _parse_table_content(self, body: str) -> Tuple[List[str], List[List[str]]]:
        """Parse markdown table content into headers and rows."""
        return parse_markdown_table(body)

    def _build_kv_section(
        self, title_uc: str, role: SectionRole, body: str
    ) -> Optional["Section"]:
        """
        Build a KVList Section if bullet kvs exist and allowed by kv_section_titles.
        """
        key = self._norm_title_key(title_uc)
        if self.kv_section_titles and key not in self.kv_section_titles:
            return None
        kv_items = parse_bullets_to_kvlist(body)
        if not kv_items:
            return None
        return Section(
            title=self._as_title(title_uc),
            role=role,
            blocks=[KVList(type="kvlist", items=kv_items)],
        )

    def _iter_sections(self, md: str) -> List[Tuple[str, str]]:
        """
        Parse markdown into an ordered list of (TITLE_UC, BODY) pairs,
        preserving duplicates and empty titles for validation upstream.
        """
        text = (md or "").strip()
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
