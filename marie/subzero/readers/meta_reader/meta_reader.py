from pprint import pprint
from typing import Any, List, Optional, Union

import numpy as np

from marie.subzero.models.models import LineModel, WordModel
from marie.subzero.readers.base import BaseReader
from marie.subzero.structures.line_metadata import LineMetadata
from marie.subzero.structures.line_with_meta import LineWithMeta
from marie.subzero.structures.unstructured_document import UnstructuredDocument


class MetaReader(BaseReader):
    """
    This reader allows handling of Metadata from marie
    """

    def __init__(self, *, config: Optional[dict] = None) -> None:
        super().__init__(config=config)

    def read(
        self, src: Union[str, dict], parameters: Optional[dict] = None
    ) -> UnstructuredDocument:
        parameters = {} if parameters is None else parameters

        return None

    def __get_text(self, value: Any) -> str:  # noqa
        if isinstance(value, (dict, list)) or value is None:
            return ""

        return str(value)

    @classmethod
    def from_data(
        cls, frames: List[np.ndarray], ocr_meta: List[dict]
    ) -> UnstructuredDocument:
        assert len(frames) == len(ocr_meta)
        unstructured_lines = []
        page_id = 0

        for frame, frame_meta in zip(frames, ocr_meta):
            lines = frame_meta["meta"]["lines"]
            unique_line_ids = sorted(np.unique(lines))
            line_bboxes = frame_meta["meta"]["lines_bboxes"]
            assert len(unique_line_ids) == len(line_bboxes)

            print(f"Unique Line IDs : {len(unique_line_ids)}")
            print(f"Line BBoxes : {len(line_bboxes)}")

            for line_idx in unique_line_ids:
                pprint(line_idx)
                lines_bbox = line_bboxes[line_idx - 1]

                meta_line = [
                    LineModel(**m_line)
                    for m_line in frame_meta["lines"]
                    if m_line["line"] == line_idx
                ][0]
                meta_words = [
                    WordModel(**word)
                    for word in frame_meta["words"]
                    if word["line"] == line_idx
                ]
                meta_line.words = meta_words
                # convert to list of WordModel
                print(f"Line / Words : {line_idx}, {len(meta_words)}")
                print(meta_line)

                data = meta_line.model_dump()

                lmd = LineMetadata(page_id=page_id, line_id=line_idx, model=meta_line)

                lwm = LineWithMeta(
                    line=meta_line.text,
                    annotations=[],
                    metadata=lmd,
                )
                print(lwm)

                unstructured_lines.append(lwm)
            # pprint(result)
            # meta = result.get("meta", {})
            # pprint(meta)
            # lines.append(cls.transform(frame, result))

        return UnstructuredDocument(lines=unstructured_lines, metadata={})

    @classmethod
    def transform(cls, frame, result) -> LineWithMeta:
        print("00000000000000000000000000000000000")
        pprint(result)

        return LineWithMeta(
            line="",
            annotations=[],
        )
