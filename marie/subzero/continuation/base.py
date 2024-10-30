from abc import ABC, abstractmethod
from typing import List, Optional

from marie.subzero.models.definition import ExecutionContext, Layer
from marie.subzero.models.match import MatchSection


class ContinuationStrategy(ABC):
    """
    Class representing the strategy for finding the continuation within a specific layer and context.
    """

    @abstractmethod
    def find_continuation(
        self,
        context: ExecutionContext,
        layer: Layer,
        matched_sections: List["MatchSection"],
        parent_layer: Optional[Layer],
    ):
        """
        :param context: The execution context which contains runtime information.
        :param layer: The layer instance within which to find the continuation.
        :param matched_sections: List of matched sections that need to be processed for continuation.
        :param parent_layer: The parent layer of the current layer.
        :return: None
        """
        pass
