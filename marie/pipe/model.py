from typing import Optional

from pydantic import BaseModel


class ClassificationResult(BaseModel):
    page: int
    classification: Optional[str]
    score: float
    classifier: Optional[str]
