from marie.extract.annotators.base import AnnotatorCapabilities, DocumentAnnotator
from marie.extract.structures.unstructured_document import UnstructuredDocument


class ValueExtractorAnnotator(DocumentAnnotator):
    """
    Annotator responsible for extracting specific values (like dates, money, etc.) from the document.
    """

    def __init__(self):
        """
        Initialize the annotator with a specific value type to extract.
        """
        super().__init__()
        self.value_type = "XYZ"  # e.g., "date", "amount"

    @property
    def capabilities(self) -> list:
        """
        Value extractor supports the EXTRACTOR capability.
        """
        return [AnnotatorCapabilities.EXTRACTOR]

    def annotate(self, document: UnstructuredDocument):
        """
        Perform value extraction on the given document.
        """
        self.validate_document(document)
        print(f"[ValueExtractor] Extracting '{self.value_type}' from document...")
        # Mocked raw output from a model
        raw_output = f"Extracted {self.value_type} values: [value1, value2]"
        return self.parse_output(raw_output)

    def parse_output(self, raw_output):
        """
        Parse the raw output from value extraction into structured data.
        """
        print("[ValueExtractor] Parsing raw model output...")
        return {"type": self.value_type, "values": ["value1", "value2"]}
