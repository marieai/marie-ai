from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel


class ClassifierBase(BaseModel):
    page: int
    score: Optional[float]
    classifier: Optional[str]
    classification: Optional[str]


class SubClassifier(BaseModel):
    classifier: str
    details: List[ClassifierBase] = None


class ClassificationResult(ClassifierBase):
    sub_classifier: List[SubClassifier] = None
