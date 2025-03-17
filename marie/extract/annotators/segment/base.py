from marie.extract.annotators.base import AnnotatorCapabilities, DocumentAnnotator
from marie.extract.structures.unstructured_document import UnstructuredDocument


class SegmentAnnotator(DocumentAnnotator):
    """
    Annotator responsible for segmenting a document into sections or logical parts.

    A segmenter annotator is expected to mark the row in the `UnstructuredDocument` where a new segment begins or ends.
    Example of a segment: Table Header, Table Body, Table Footer, etc.
    """

    def __init__(self):
        """
        Initialize the segment annotator
        """
        super().__init__()

    @property
    def capabilities(self) -> list:
        """
        Segment annotator supports the SEGMENTER capability.
        """
        return [AnnotatorCapabilities.SEGMENTER]

    def annotate(self, document: UnstructuredDocument):
        """
        Segment the given document into logical parts.
        """
        self.validate_document(document)
        print(f"[SegmentAnnotator] Splitting document ")
        # Mocked raw output from a model
        raw_output = ["Segment 1", "Segment 2", "Segment 3"]

        return self.parse_output(raw_output)

    def parse_output(self, raw_output):
        """
        Parse raw model segmentation output into structured data.
        """
        print("[SegmentAnnotator] Parsing raw segmentation output...")
        return [
            {"segment_id": i + 1, "content": segment}
            for i, segment in enumerate(raw_output)
        ]
