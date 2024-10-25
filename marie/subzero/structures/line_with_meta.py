from typing import List, Optional, Sized, Union

from pydantic import BaseModel

from marie.subzero.structures.annotation import Annotation
from marie.subzero.structures.line_metadata import LineMetadata
from marie.subzero.structures.serializable import Serializable


class LineWithMeta(Sized, Serializable):
    """
    Structural unit of document - line (or paragraph) of text and its metadata.
    One LineWithMeta should not contain text from different logical parts of the document
    (for example, document title and raw text of the document should not be in the same line).
    Still the logical part of the document may be represented by more than one line (for example, document title may consist of many lines).
    """

    def __init__(
        self,
        line: str,
        metadata: Optional[LineMetadata] = None,
        annotations: Optional[List[Annotation]] = None,
        uid: str = None,
    ) -> None:
        """
        :param line: raw text of the document line
        :param metadata: metadata (related to the entire line, as line or page number, its hierarchy level)
        :param annotations: metadata that refers to some part of the text, for example, font size, font type, etc.
        :param uid: unique identifier of the line
        """
        from uuid import uuid1

        self._line = line
        self._metadata = (
            LineMetadata(page_id=0, line_id=None) if metadata is None else metadata
        )
        self._annotations = [] if annotations is None else annotations
        self._uid = str(uuid1()) if uid is None else uid

    def __len__(self) -> int:
        return len(self._line)

    def __lt__(self, other: "LineWithMeta") -> bool:
        return self.line < other.line

    def __getitem__(self, index: Union[slice, int]) -> "LineWithMeta":
        if isinstance(index, int):
            if len(self) == 0 or index >= len(self) or index < -len(self):
                raise IndexError("Get item on empty line")
            index %= len(self)
            return self[index : index + 1]
        if isinstance(index, slice):
            start = index.start if index.start else 0
            stop = index.stop if index.stop is not None else len(self)
            step = 1 if index.step is None else index.step
            if start < 0 or stop < 0 or step != 1:
                raise NotImplementedError()
            if start > len(self) or 0 < len(self) == start:
                raise IndexError("start > len(line)")

            annotations = self.__extract_annotations_by_slice(start, stop)
            return LineWithMeta(
                line=self.line[start:stop],
                metadata=self.metadata,
                annotations=annotations,
            )
        else:
            raise TypeError("line indices must be integers")

    def __extract_annotations_by_slice(self, start: int, stop: int) -> List[Annotation]:
        """
        Extract annotations for given slice.
        """
        assert start >= 0
        assert stop >= 0
        annotations = []
        for annotation in self.annotations:
            if start < annotation.end and stop > annotation.start:
                annotations.append(
                    Annotation(
                        start=max(annotation.start, start) - start,
                        end=min(annotation.end, stop) - start,
                        name=annotation.name,
                        value=annotation.value,
                        bboxes=annotation.bboxes,
                    )
                )
        return annotations

    @property
    def line(self) -> str:
        """
        Raw text of the document line
        """
        return self._line

    @property
    def metadata(self) -> LineMetadata:
        """
        Line metadata related to the entire line, as line or page number, hierarchy level
        """
        return self._metadata

    @property
    def annotations(self) -> List[Annotation]:
        """
        Metadata that refers to some part of the text, for example, font size, font type, etc.
        """
        return self._annotations

    @property
    def uid(self) -> str:
        """
        Unique identifier of the line
        """
        return self._uid

    def set_line(self, line: str) -> None:
        self._line = line

    def set_metadata(self, metadata: LineMetadata) -> None:
        self._metadata = metadata

    def __repr__(self) -> str:
        text = self.line if len(self.line) < 65 else self.line[:62] + "..."
        return f"LineWithMeta({text.strip()})"

    def __add__(self, other: Union["LineWithMeta", str]) -> "LineWithMeta":
        from marie.subzero.annotation_merger import AnnotationMerger

        assert isinstance(other, (LineWithMeta, str))
        if len(other) == 0:
            return self
        if isinstance(other, str):
            line = self.line + other
            return LineWithMeta(
                line=line,
                metadata=self._metadata,
                annotations=self.annotations,
                uid=self.uid,
            )
        line = self.line + other.line
        shift = len(self)
        other_annotations = []
        for annotation in other.annotations:
            new_annotation = Annotation(
                start=annotation.start + shift,
                end=annotation.end + shift,
                name=annotation.name,
                value=annotation.value,
            )
            other_annotations.append(new_annotation)
        annotations = AnnotationMerger().merge_annotations(
            self.annotations + other_annotations, text=line
        )
        return LineWithMeta(
            line=line, metadata=self._metadata, annotations=annotations, uid=self.uid
        )

    def to_model(self) -> BaseModel:
        raise NotImplementedError

    @classmethod
    def join(cls, lines, delimiter):
        raise NotImplementedError
