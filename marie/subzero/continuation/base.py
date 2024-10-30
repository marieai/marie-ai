from abc import ABC, abstractmethod
from typing import List, Optional

from marie.subzero.models.definition import ExecutionContext, Layer
from marie.subzero.models.results import MatchSection


class ContinuationStrategy(ABC):

    @abstractmethod
    def find_continuation(
        self,
        context: ExecutionContext,
        layer: Layer,
        matched_sections: List['MatchSection'],
    ):
        pass
