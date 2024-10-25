from typing import List

from marie.models.pix2pix.models import BaseModel
from marie.subzero.structures.annotation import Annotation
from marie.subzero.structures.line_with_meta import LineWithMeta
from marie.subzero.structures.serializable import Serializable


class CellWithMeta(Serializable):
    """
    This class holds the information about the cell: list of lines and cell properties (rowspan, colspan, invisible).

    """

    def __init__(
        self,
        lines: List[LineWithMeta],
        colspan: int = 1,
        rowspan: int = 1,
        invisible: bool = False,
    ) -> None:
        """
        :param lines: textual lines of the cell
        :param colspan: number of columns to span like in HTML format
        :param rowspan: number of rows to span like in HTML format
        :param invisible: indicator for displaying or hiding cell text
        """
        self.lines: List[LineWithMeta] = lines
        self.colspan: int = colspan
        self.rowspan: int = rowspan
        self.invisible: bool = invisible

    def __repr__(self) -> str:
        return f"CellWithMeta({self.get_text()[:65]})"

    def get_text(self) -> str:
        """
        Get merged text of all cell lines
        """
        return "\n".join([line.line for line in self.lines])

    def get_annotations(self) -> List[Annotation]:
        """
        Get merged annotations of all cell lines (start/end of annotations moved according to the merged text)
        """
        return LineWithMeta.join(lines=self.lines, delimiter="\n").annotations

    @staticmethod
    def create_from_cell(cell: "CellWithMeta") -> "CellWithMeta":
        return CellWithMeta(
            lines=cell.lines,
            colspan=cell.colspan,
            rowspan=cell.rowspan,
            invisible=cell.invisible,
        )

    def to_model(self) -> BaseModel:
        raise NotImplementedError
