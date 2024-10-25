from typing import List, Optional

from pydantic import BaseModel, conlist


class WordModel(BaseModel):
    id: int
    text: str
    confidence: float
    box: List[int]
    line: int
    word_index: int


class LineModel(BaseModel):
    line: int
    wordids: List[int]
    text: str
    bbox: List[int]
    confidence: float
    words: Optional[List[WordModel]] = None
