from typing import List, Optional, Tuple

from docarray.base_doc.doc import BaseDoc, BaseDocWithoutId
from pydantic import BaseModel


class TemplateMatchResult(BaseModel):
    bbox: Tuple[int, int, int, int]
    label: str
    score: float
    similarity: float
    frame_index: Optional[int] = 0


class TemplateMatchResultDoc(BaseDocWithoutId, frozen=True):
    bbox: Tuple[int, int, int, int]
    label: str
    score: float
    similarity: float
    frame_index: Optional[int] = 0


class TemplateMatchingResultDoc(BaseDoc, frozen=True):
    asset_key: str
    results: List[TemplateMatchResultDoc]


class TemplateSelector(BaseDocWithoutId, frozen=True):
    region: List[int]
    frame: str
    bbox: List[int]
    label: str
    text: str
    create_window: bool
    top_k: int


class TemplateMatchingRequestDoc(BaseDoc):
    asset_key: str
    id: str
    pages: List[int]
    score_threshold: float
    scoring_strategy: str
    max_overlap: float
    window_size: List[int]
    matcher: str
    downscale_factor: float
    selectors: List[TemplateSelector]
