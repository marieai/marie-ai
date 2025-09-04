import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from rtree import index

import marie.check as check
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.structured_region import StructuredRegion, pagespan_pages
from marie.extract.structures.table import Table


class UnstructuredDocument:
    """
    A document that contains a list of lines with metadata.
    """

    def __init__(
        self,
        lines: List[LineWithMeta],
        regions: Optional[List[StructuredRegion]],
        metadata: Dict[str, Any],
    ) -> None:
        self.metadata = metadata
        # this original metadata from source *.meta.json
        self.source_metadata = (
            metadata.get("source_metadata", None) if metadata else None
        )
        if not self.source_metadata:
            raise ValueError("source_metadata is required in the metadata")

        self.lines: List[LineWithMeta] = lines
        self.regions: List[StructuredRegion] = regions if regions is not None else []

        self.rtree_by_page = {}
        self.insert(lines)
        self._lines_by_page = defaultdict(list)
        # FIXME: this is not correct, we currently assume that all pages are present. but this is not always the case
        self._page_count = int(self.source_metadata.get("pages", 0))
        # Initialize an empty list for each expected page to ensure contiguous page ID mapping,
        # even if some pages are blank.
        for page_id in range(self._page_count):
            self._lines_by_page[page_id] = []

        for line in lines:
            self._lines_by_page[line.metadata.page_id].append(line)

    @property
    def lines_by_page(self) -> Dict[int, List[LineWithMeta]]:
        """
        Retrieve lines grouped by their page ID.
        Returns:
            Dict[int, List[LineWithMeta]]: A dictionary mapping page IDs to their associated lines.
        """
        return self._lines_by_page

    def insert(self, lines: List[LineWithMeta]) -> None:

        if True:
            return None
        id_set = set()
        for i, line in enumerate(lines):
            page_id = line.metadata.page_id
            words = line.metadata.model.words
            rtree = self.get_or_create_rtree(page_id)

            for word in words:
                if word.id in id_set:
                    raise ValueError(
                        f"Word with id {word.id} already exists in the document, cannot insert it twice."
                    )
                id_set.add(word.id)
                rtree.insert(word.id, word.to_xyxy(), obj=word)

    def __str__(self) -> str:
        return f"UnstructuredDocument with {self.page_count} pages, {len(self.lines)} lines and metadata keys: {list(self.metadata.keys())}"

    def get_or_create_rtree(self, page_id: int) -> index.Index:
        if page_id not in self.rtree_by_page:
            self.rtree_by_page[page_id] = index.Index()
        return self.rtree_by_page[page_id]

    def query(self, page_id: int, query_bbox: List[int]) -> List[LineWithMeta]:
        """
        Query the R-tree for bounding boxes that intersect with a given area on a specific page.
        """
        rtree = self.get_or_create_rtree(page_id)
        results = list(rtree.intersection((page_id, *query_bbox), objects=True))
        return [result.object for result in results]

    def knn_search(
        self, page_id: int, query_point: List[int], k: int
    ) -> List[LineWithMeta]:
        """
        Perform k-nearest neighbors search for a given point on a specific page.
        bounding box format (minx, miny, maxx, maxy) where minx and maxx are the x-coordinate, and miny and maxy are the y-coordinate of the query point.
        """
        rtree = self.get_or_create_rtree(page_id)
        results = list(
            rtree.nearest(
                (query_point[0], query_point[1], query_point[0], query_point[1]),
                k,
                objects=True,
            )
        )
        return [result.object for result in results]

    @property
    def page_count(self) -> int:
        # return len(set(line.metadata.page_id for line in self.lines))
        return self._page_count

    def to_text(
        self,
        collapsed_text: bool = False,
        decorator: Optional[Callable[[str, int], str]] = None,
        page_number: Optional[int] = None,
    ) -> str:
        """
        Convert document lines to a text string, grouped by page.

        :param collapsed_text: Collapse multiple empty lines into one (True) or leave as-is (False).
        :param decorator: A function applied to each line’s text (if provided).
        :param page_number: If provided, only lines from this page are converted. Otherwise, all pages are included.
        :return: Combined text of the document (one or all pages) with optional modifications.
        """

        lines_by_page = self.lines_by_page
        if page_number is not None:
            if page_number in lines_by_page:
                pages_to_process = [page_number]
            else:
                return ""
        else:
            pages_to_process = sorted(
                lines_by_page.keys(), key=lambda x: (x is None, x)
            )

        # Build text for each page, then combine
        all_pages_text = []
        for pid in pages_to_process:
            sorted_lines = sorted(
                lines_by_page[pid], key=lambda ln: ln.metadata.line_id
            )

            text_lines = []
            for row_number, ln_with_meta in enumerate(sorted_lines):
                line_text = ln_with_meta.line
                if decorator:
                    line_text = decorator(line_text, ln_with_meta.metadata.line_id)
                else:
                    line_text = line_text.strip()
                text_lines.append(line_text)

            page_text = "\n".join(text_lines)
            all_pages_text.append(page_text)

        combined_text = "\n\n".join(all_pages_text)

        #  collapse extra blank lines
        if collapsed_text:
            combined_text = re.sub(r'\n\s*\n+', '\n', combined_text)

        return combined_text

    def lines_for_page(self, page_id: int) -> List[LineWithMeta]:
        """
        Retrieve all lines belonging to the specified page_id.
        :param page_id: The page ID for which to retrieve lines.
        :return: A list of LineWithMeta objects associated with the specified page ID.
        """
        check.int_param(page_id, "page_id")
        return self._lines_by_page[page_id]

    def tables_for_page(self, page_id: int) -> List[Table]:
        """
        Retrieve all tables belonging to the specified page_id.

        :param page_id: The page ID for which to retrieve tables.
        :return: A list of Table objects associated with the specified page ID.
        """
        raise RuntimeError('Method has been deprecated and is being removed.')

    def regions_for_page(self, page_id: int) -> List[StructuredRegion]:
        """
        Retrieve all regions belonging to the specified page_id.

        :param page_id: The page ID for which to retrieve regions.
        :return: A list of StructuredRegion objects associated with the specified page ID.
        """
        check.int_param(page_id, "page_id")
        return [
            region
            for region in self.regions
            if region.span and page_id in pagespan_pages(region.span)
        ]

    @property
    def page_ids(self) -> List[int]:
        """
        Retrieve all unique page IDs from an UnstructuredDocument instance, sorted in ascending order.
        Returns:
            List[int]: A sorted list of unique page IDs.
        """
        return sorted(set(line.metadata.page_id for line in self.lines))

    def insert_table(self, table: Table) -> None:
        """
        Insert a new table into the document.

        :param table: Table object to be added to the document.
        """
        if not isinstance(table, Table):
            raise ValueError("The object to insert must be an instance of Table.")

        self.tables.append(table)

    def insert_region(self, region: StructuredRegion) -> None:
        """
        Insert a new region into the document.

        :param region: StructuredRegion object to be added to the document.
                      The region must have 'source' and 'source_layer' tags set
                      for proper traceability.
        :raises ValueError: If region is not a StructuredRegion instance or if
                           required traceability tags are missing.
        """
        if not isinstance(region, StructuredRegion):
            raise ValueError(
                "The object to insert must be an instance of StructuredRegion."
            )

        # Validate that required tags are present for traceability
        if "source" not in region.tags:
            raise ValueError(
                f"Region {region.region_id or 'unknown'} is missing required 'source' tag for traceability"
            )

        self.regions.append(region)
