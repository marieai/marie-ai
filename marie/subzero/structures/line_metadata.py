from typing import Dict, Optional, Union

from pydantic import BaseModel

from marie.subzero.structures.serializable import Serializable


class LineMetadata(Serializable):
    """
    This class holds information about document node (and document line) metadata, such as page number or line level in a document hierarchy.
    """

    def __init__(
        self,
        page_id: int,
        line_id: Optional[int],
        **kwargs: Dict[str, Union[str, int, float]],
    ) -> None:
        """
        :param page_id: page number where paragraph starts, the numeration starts from page 0
        :param line_id: line number inside the entire document, the numeration starts from line 0
        """
        self.page_id: int = page_id
        self.line_id: Optional[int] = line_id
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        metadata = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
        return f"LineMetadata(page_id={self.page_id}, line_id={self.line_id}, metadata={metadata})"

    def to_model(self) -> BaseModel:
        raise NotImplementedError
