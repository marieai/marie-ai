from abc import ABC, abstractmethod
from typing import List

from marie.subzero.models.definition import ExecutionContext, FieldMapping
from marie.subzero.models.results import MatchField, MatchSection


class FieldExtractionHandler(ABC):
    """
    Field extraction handler interface
    """

    @abstractmethod
    def can_handle(self, mapping: FieldMapping) -> bool:
        """
        Check if the implementor can handle given field mapping

        :param mapping: FieldMapping to check
        :return: boolean indicating if the handler can handle the given field mapping
        """
        pass

    @abstractmethod
    def handle(
        self,
        context: ExecutionContext,
        mapping: FieldMapping,
        target_section: MatchSection,
        match_section: MatchSection,
    ) -> List[MatchField]:
        """
        Extract data

        :param context: ExecutionContext to use
        :param mapping: FieldMapping to use for extraction
        :param target_section: This is the upper and lower bound where data can reside
        :param match_section: This is the original cutpoint without any adjustments
        :return: List of MatchField
        """
        pass
