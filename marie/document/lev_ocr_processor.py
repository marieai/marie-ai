import os
import typing

from marie.document.ocr_processor import OcrProcessor
from marie.models.icr.model import Model
from marie.constants import __model_path__


class LevenshteinOcrProcessor(OcrProcessor):
    def __init__(
        self,
        work_dir: str = "/tmp/icr",
        models_dir: str = os.path.join(__model_path__, "levocr"),
        cuda: bool = True,
    ) -> None:
        super().__init__(work_dir, cuda)
        print("LevenshteinOCR processor [cuda={}]".format(cuda))

    def recognize_from_fragments(
        self, images, **kwargs
    ) -> typing.List[typing.Dict[str, any]]:
        """Recognize text from image fragments

        Args:
            images: A list of input images, supplied as numpy arrays with shape
                (H, W, 3).
        """

        raise NotImplementedError("Levenshtein OCR is not implemented yet")

    def is_available(self) -> bool:
        return False
