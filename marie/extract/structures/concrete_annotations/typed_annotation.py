from typing import Any, Dict, Union

from pydantic import BaseModel

from marie.extract.structures.annotation import Annotation


class TypedAnnotation(Annotation):

    def __init__(
        self,
        start: int,
        end: int,
        name: str,
        value: Union[str, Dict[str, Any]],
        annotation_type: str,
        bboxes: [],
    ):
        super().__init__(start, end, name, value, bboxes)
        self.annotation_type = annotation_type.upper()

    def to_model(self) -> BaseModel:
        pass
