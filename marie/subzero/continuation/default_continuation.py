from typing import List, Optional

from marie.subzero.continuation.base import ContinuationStrategy
from marie.subzero.models.definition import ExecutionContext, Layer
from marie.subzero.models.match import MatchSection


class DefaultContinuationStrategy(ContinuationStrategy):
    def find_continuation(
        self,
        context: ExecutionContext,
        layer: Layer,
        matched_sections: List[MatchSection],
        parent_layer: Optional[Layer],
    ):
        pass
