import logging
from collections import namedtuple
from typing import List, Tuple

from marie.subzero.exceptions import GrapnelException
from marie.subzero.extractor.field.base import BaseFieldExtractionHandler
from marie.subzero.models.definition import (
    CompositeFieldMapping,
    Dimension,
    ExecutionContext,
    FieldMapping,
    Margin,
    Perimeter,
)
from marie.subzero.models.match import MatchField, MatchSection
from marie.subzero.processor.scan_result import GrapnelScanResult
from marie.subzero.processor.selector_matcher import SelectorMatcher
from marie.subzero.utils.blob_util import BlobUtil
from marie.subzero.utils.conversion_util import ConversionUtil

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
