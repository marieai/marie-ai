from typing import Optional, Tuple

from pydantic import BaseModel


class TemplateMatchResult(BaseModel):
    bbox: Tuple[int, int, int, int]
    label: str
    score: float
    similarity: float
    frame_index: Optional[int] = 1
