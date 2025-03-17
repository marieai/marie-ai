import logging
from collections import defaultdict
from typing import Dict, List

from marie.extract.extractor.row.row_extraction_handler import RowExtractionHandler
from marie.extract.models.base import Blob
from marie.extract.models.definition import (
    ExecutionContext,
    FieldMapping,
    Layer,
    RowExtractionStrategy,
)
from marie.extract.models.match import MatchFieldRow, MatchSection, Span

LOGGER = logging.getLogger(__name__)


class CompositeRowExtractionHandler(RowExtractionHandler):
    """
    Composite Row Layout handler
    If there is no primary column defined in the FieldMapping then we will handle row extraction
    based on the Ordinal columns, Ordinals are 0 based, if we need to skip row then we need to place a 'dummy ordinal
    field' in that row
    This strategy is the simplest and it does not require any changes to the UI system.
    """

    def can_handle(self, section: MatchSection, mapping: List[FieldMapping]) -> bool:
        """
        Check if the implementor can handle given field mappings

        :param section: MatchSection to check
        :param mapping: List of FieldMapping to check
        :raises GrapnelException: when chosen strategy does not match selected parameters
        :return: boolean indicating if the handler can handle the given field mappings
        """
        res = section.row_extraction_strategy
        if res == RowExtractionStrategy.COMPOSITE_FIXED:
            primary_column_index = self.find_primary_column_indexes(mapping)
            if primary_column_index:
                raise ValueError(
                    f"Selected Strategy '{res}' does not have all the required values set properly, Expected No Primary Columns but found one at {primary_column_index}"
                )
            return True
        return False

    def extract(
        self,
        context: ExecutionContext,
        section: MatchSection,
        span: Span,
        rows: List[List[Blob]],
        field_mappings: List[FieldMapping],
        layer: Layer,
    ) -> List[MatchFieldRow]:
        """
        Perform row extraction

        :param context: ExecutionContext to use
        :param section: MatchSection to match against
        :param span: Span location where the results are to be extracted from
        :param rows: List of collected data
        :param field_mappings: List of FieldMapping used for data extraction
        :param layer: Layer to use
        :return: List of MatchFieldRow
        """
        if not rows:
            return []

        pass
