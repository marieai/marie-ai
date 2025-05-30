import logging
from collections import namedtuple
from typing import List, Tuple

from marie.extract.extractor.field.base import BaseFieldExtractionHandler
from marie.extract.models.definition import (
    CompositeFieldMapping,
    Dimension,
    ExecutionContext,
    FieldMapping,
    Margin,
    Perimeter,
)
from marie.extract.models.match import MatchField, MatchSection

LOGGER = logging.getLogger(__name__)

CompositeMatch = namedtuple('CompositeMatch', ['lhs', 'rhs'])


class CompositeFieldExtractionHandler(BaseFieldExtractionHandler):
    """
    Extractor for composite fields that do not contain primary fields
    """

    def __init__(self):
        super().__init__()
        LOGGER.info("CompositeFieldExtractionHandler initialized")

    def can_handle(self, mapping: FieldMapping) -> bool:
        """
        Check if the implementor can handle given field mapping

        :param mapping: FieldMapping to check
        :return: boolean indicating if the handler can handle the given field mapping
        """
        return isinstance(mapping, CompositeFieldMapping)

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
