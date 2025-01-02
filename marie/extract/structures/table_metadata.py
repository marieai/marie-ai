from typing import Optional

from pydantic import BaseModel

from marie.extract.structures.serializable import Serializable


class TableMetadata(Serializable):
    """
    This class holds the information about table unique identifier, rotation angle (if table has been rotated - for images) and so on.

    """

    def to_model(self) -> BaseModel:
        pass

    def __init__(
        self,
        page_id: Optional[int],
        uid: Optional[str] = None,
        rotated_angle: float = 0.0,
        title: str = "",
    ) -> None:
        """
        :param page_id: number of the page where table starts
        :param uid: unique identifier of the table
        :param rotated_angle: rotation angle by which the table was rotated during recognition
        :param title: table's title
        """
        import uuid

        self.page_id: Optional[int] = page_id
        self.uid: str = str(uuid.uuid4()) if not uid else uid
        self.rotated_angle: float = rotated_angle
        self.title: str = title
