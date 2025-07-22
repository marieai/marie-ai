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

    def rows(self) -> List[List[CellWithMeta]]:
        """
        Get the cells of the table.
        :return: A list of lists of cells.
        """
        return self.cells

    @staticmethod
    def format_cell(cell: CellWithMeta) -> str:
        if cell is None:
            return ""
        lines = cell.lines
        if not lines:
            return ""
        return "\n".join([line.line for line in lines])

    def __str__(self) -> str:
        try:

            from prettytable import PrettyTable

            table = PrettyTable()

            if not self.cells:
                return "Empty table"

            headers = [Table.format_cell(cell) for cell in self.cells[0]]
            # Ensure the headers are unique before assigning them to table.field_names
            unique_headers = []
            seen_headers = set()

            for header in headers:
                if header in seen_headers:
                    counter = 1
                    new_header = f"{header}_{counter}"
                    while new_header in seen_headers:
                        counter += 1
                        new_header = f"{header}_{counter}"
                    unique_headers.append(new_header)
                    seen_headers.add(new_header)
                else:
                    unique_headers.append(header)
                    seen_headers.add(header)

            table.field_names = unique_headers

            # Add remaining rows as data
            for row in self.cells[1:]:
                table.add_row(
                    [
                        Table.format_cell(cell) if cell is not None else ""
                        for cell in row
                    ]
                )

            return str(table)
        except Exception as e:
            return f"Error generating table representation: {e} \n{self.cells}"
