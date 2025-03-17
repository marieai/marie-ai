import logging
from typing import List

from marie.extract.extractor.field.base import BaseFieldExtractionHandler
from marie.extract.models.definition import ExecutionContext, FieldMapping
from marie.extract.models.match import MatchField, MatchSection

LOGGER = logging.getLogger(__name__)


class SelectorFieldExtractionHandler(BaseFieldExtractionHandler):
    """
    Selector Field Extraction Handler
    """

    def __init__(self):
        super().__init__()
        LOGGER.info("SelectorFieldExtractionHandler initialized")

    def can_handle(self, mapping: FieldMapping) -> bool:
        """
        Check if the implementor can handle given field mapping

        :param mapping: FieldMapping to check
        :return: boolean indicating if the handler can handle the given field mapping
        """
        return mapping.selector is not None

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
