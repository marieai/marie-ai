import logging
from typing import List

from marie.extract.extractor.row_extraction_handler import RowExtractionHandler
from marie.extract.models.base import Blob
from marie.extract.models.definition import (
    ExecutionContext,
    FieldMapping,
    Layer,
    RowExtractionStrategy,
)
from marie.extract.models.match import MatchFieldRow, MatchSection, Span

LOGGER = logging.getLogger(__name__)


class PrimaryRowExtractionHandler(RowExtractionHandler):
    """
    Primary Row Extraction handler that specifies how rows will be extracted for given FieldMapping
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
        if res in [
            RowExtractionStrategy.PRIMARY_COLUMN_VARIABLE,
            RowExtractionStrategy.PRIMARY_COLUMN_FIXED,
        ]:
            primary_column_indexes = self.find_primary_column_indexes(mapping)
            if not primary_column_indexes:
                raise ValueError(
                    f"Selected Strategy '{res}' does not have all the required values set properly, Expected Primary Column but none were found"
                )
            return True
        return False

    def extract(
        self,
        context: ExecutionContext,
        section: MatchSection,
        span: Span,
        rows: List[List[Blob]],
        field_mappings_source: List[FieldMapping],
        layer: Layer,
    ) -> List[MatchFieldRow]:
        """
        Perform row extraction

        :param context: ExecutionContext to use
        :param section: MatchSection to match against
        :param span: Span location where the results are to be extracted from
        :param rows: List of collected data
        :param field_mappings_source: List of FieldMapping used for data extraction
        :param layer: Layer to use
        :return: List of MatchFieldRow
        """
        pass
