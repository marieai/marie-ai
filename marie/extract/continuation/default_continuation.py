from typing import List, Optional

from marie.extract.continuation.base import ContinuationStrategy
from marie.extract.models.definition import ExecutionContext, Layer
from marie.extract.models.match import MatchSection


class DefaultContinuationStrategy(ContinuationStrategy):
    def find_continuation(
        self,
        context: ExecutionContext,
        layer: Layer,
        matched_sections: List[MatchSection],
        parent_layer: Optional[Layer],
    ):
        pass
