from typing import List

from pydantic import BaseModel

from marie.extract.structures.cell_with_meta import CellWithMeta
from marie.extract.structures.serializable import Serializable
from marie.extract.structures.table_metadata import TableMetadata


class Table(Serializable):
    """
    This class holds information about tables in the document.
    We assume that a table has rectangle form (has the same number of columns in each row).
    If some cells are merged, they are duplicated and information about merge is stored in rowspan and colspan.
    Table representation is row-based i.e. external list contains list of rows.

    """

    def __init__(
        self, cells: List[List[CellWithMeta]], metadata: TableMetadata
    ) -> None:
        """
        :param cells: a list of lists of cells
        :param metadata: table metadata
        """
        self.metadata: TableMetadata = metadata
        self.cells: List[List[CellWithMeta]] = cells

    def to_model(self) -> BaseModel:
        cells = [[cell.to_model() for cell in row] for row in self.cells]
        return None
        # return ApiTable(cells=cells, metadata=self.metadata.to_api_schema())
