from typing import Any, Dict, List

from rtree import index

from marie.subzero.structures.line_with_meta import LineWithMeta


class UnstructuredDocument:
    """
    A document that contains a list of lines with metadata.
    """

    def __init__(self, lines: List[LineWithMeta], metadata: Dict[str, Any]) -> None:
        self.metadata = metadata
        self.lines = lines
        self.rtree_by_page = {}
        self.insert(lines)

    def insert(self, lines: List[LineWithMeta]) -> None:

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
