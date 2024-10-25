import re
from typing import List

from marie.subzero.structures.annotation import Annotation


class Space:

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end


class AnnotationMerger:
    spaces = re.compile(r"\s+")

    def merge_annotations(
        self, annotations: List[Annotation], text: str
    ) -> List[Annotation]:
        """
        Merge annotations when end of the first annotation and start of the second match and has same value.
        Used with add_text
        """
        return []
