from abc import ABC
from typing import Optional

from marie.subzero.models.base import Dimension, Perimeter
from marie.subzero.models.definition import FieldMapping
from marie.subzero.models.results import MatchField, ResultType, ScanResult


class BaseFieldExtractionHandler(ABC):
    """
    Abstract base class for field extraction handlers.
    """

    @staticmethod
    def create_match_field(
        page_index: int, mapping: FieldMapping, y: int, x_offset: int
    ) -> MatchField:
        """
        Create a MatchField for the given FieldMapping.

        :param page_index: Index of the page
        :param mapping: FieldMapping to use
        :param y: Y coordinate
        :param x_offset: X offset
        :return: MatchField
        """
        perimeter: Perimeter = mapping.search_perimeter
        dimension: Dimension = mapping.dimension
        width: float = dimension.width
        x1: float = perimeter.x_min + x_offset
        gsr = ScanResult(x1, y, width, dimension.height, ResultType.BLOB)
        gsr.page = page_index
        gsr.x_offset = x_offset

        return MatchField.create(mapping.identifier, gsr)
