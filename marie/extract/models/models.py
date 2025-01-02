from typing import List, Optional

from pydantic import BaseModel


class WordModel(BaseModel):
    id: int
    text: str
    confidence: float
    box: List[int]
    line: int
    word_index: int

    def to_xyxy(self) -> List[int]:
        """
        Convert bounding box from (x, y, w, h) to (minx, miny, maxx, maxy).
        """
        x, y, w, h = self.box
        minx, miny = x, y
        maxx, maxy = x + w, y + h
        return [minx, miny, maxx, maxy]


class LineModel(BaseModel):
    line: int
    wordids: List[int]
    text: str
    bbox: List[int]
    confidence: float
    words: Optional[List[WordModel]] = None
