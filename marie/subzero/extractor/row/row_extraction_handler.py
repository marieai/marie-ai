from abc import ABC, abstractmethod
from typing import List, Set

from marie.subzero.models.base import Blob
from marie.subzero.models.definition import ExecutionContext, FieldMapping, Layer
from marie.subzero.models.match import MatchFieldRow, MatchSection, Span


class RowExtractionHandler(ABC):
    """
    Row Extraction handler that specifies how rows will be extracted for given FieldMapping
    """

    @abstractmethod
    def can_handle(self, section: MatchSection, mapping: List[FieldMapping]) -> bool:
        """
        Check if the implementor can handle given field mappings

        :param section: MatchSection to check
        :param mapping: List of FieldMapping to check
        :raises Exception: when chosen strategy does not match selected parameters
        :return: boolean indicating if the handler can handle the given field mappings
        """
        pass

    @abstractmethod
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
        pass

    @staticmethod
    def find_primary_column_indexes(field_mappings: List[FieldMapping]) -> Set[int]:
        out = set()
        for i, mapping in enumerate(field_mappings):
            if mapping.is_primary():
                out.add(i)
        return out
