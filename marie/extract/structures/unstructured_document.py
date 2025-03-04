import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from rtree import index

import marie.check as check
from marie.extract.structures.line_with_meta import LineWithMeta


class UnstructuredDocument:
    """
    A document that contains a list of lines with metadata.
    """

    def __init__(self, lines: List[LineWithMeta], metadata: Dict[str, Any]) -> None:
        self.metadata = metadata
        self.lines = lines
        self.rtree_by_page = {}
        self.insert(lines)

        self._lines_by_page = defaultdict(list)
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
        return len(set(line.metadata.page_id for line in self.lines))

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
        """
        check.int_param(page_id, "page_id")
        return self._lines_by_page[page_id]

    @property
    def page_ids(self) -> List[int]:
        """
        Retrieve all unique page IDs from an UnstructuredDocument instance, sorted in ascending order.
        Returns:
            List[int]: A sorted list of unique page IDs.
        """
        return sorted(set(line.metadata.page_id for line in self.lines))
