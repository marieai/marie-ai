from abc import abstractmethod
from typing import List

from pydantic import BaseModel


class ClassificationResult(BaseModel):
    classification: int
    score: float
    classifier: str


class VotingStrategy:
    @abstractmethod
    def vote(self, results: List[ClassificationResult]) -> tuple[str, float]:
        pass


class MajorityVoter(VotingStrategy):
    def __init__(self) -> None:
        super().__init__()

    def vote(self, results: List[ClassificationResult]) -> tuple[str, float]:
        pass


class MaxScoreVoter(VotingStrategy):
    def __init__(self) -> None:
        super().__init__()

    def vote(self, results: List[ClassificationResult]) -> tuple[str, float]:
        pass
