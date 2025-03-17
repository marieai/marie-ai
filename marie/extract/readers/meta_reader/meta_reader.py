from pprint import pprint
from typing import Any, List, Optional, Union

import numpy as np

from marie.extract.models.models import LineModel, WordModel
from marie.extract.readers.base import BaseReader
from marie.extract.structures.line_metadata import LineMetadata
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.unstructured_document import UnstructuredDocument


class MetaReader(BaseReader):
    """
    This reader allows handling of Metadata from marie-ai
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
        META_KEY = "meta"
        LINES_KEY = "lines"
        PAGE_KEY = "page"
        LINES_BBOXES_KEY = "lines_bboxes"
        WORDS_KEY = "words"

        assert len(frames) == len(ocr_meta), "Mismatch between frames and OCR metadata"

        def create_line_with_meta(meta_line, page_id, words):
            """Helper function to create a LineWithMeta instance."""
            aligned_words = [w for w in words if w["id"] in meta_line["wordids"]]
            meta_line_model = LineModel(**meta_line)
            meta_line_model.words = [WordModel(**w) for w in aligned_words]
            line_metadata = LineMetadata(
                page_id=page_id, line_id=int(meta_line["line"]), model=meta_line_model
            )
            return LineWithMeta(
                line=meta_line_model.text, annotations=[], metadata=line_metadata
            )

        unstructured_lines = []

        for frame, frame_meta in zip(frames, ocr_meta):
            # Extract relevant metadata
            meta = frame_meta[META_KEY]
            lines = meta[LINES_KEY]
            page_id = meta[PAGE_KEY]
            unique_line_ids = sorted(np.unique(lines))
            lines_bboxes = np.array(meta[LINES_BBOXES_KEY])
            frame_lines = np.array(frame_meta[LINES_KEY])
            words = np.array(frame_meta[WORDS_KEY])

            assert len(unique_line_ids) == len(
                lines_bboxes
            ), f"Unique Line IDs: {len(unique_line_ids)}, Line BBoxes: {len(lines_bboxes)}"

            for meta_line in frame_lines:
                line_with_meta = create_line_with_meta(meta_line, page_id, words)
                unstructured_lines.append(line_with_meta)

        return UnstructuredDocument(lines=unstructured_lines, metadata={})

    @classmethod
    def transform(cls, frame, result) -> LineWithMeta:
        raise NotImplementedError()
        return LineWithMeta(
            line="",
            annotations=[],
        )
