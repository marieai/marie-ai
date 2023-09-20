import os
import typing

from marie.document.ocr_processor import OcrProcessor
from marie.logging.logger import MarieLogger
from marie.constants import __model_path__


class TesseractOcrProcessor(OcrProcessor):
    """A processor which uses tesseract OCR to process"""

    def __init__(
        self,
        work_dir: str = "/tmp/icr",
        models_dir: str = os.path.join(__model_path__, "tessdata"),
        cuda: bool = True,
    ) -> None:
        super().__init__(work_dir, cuda)
        self.logger = MarieLogger(context=self.__class__.__name__)
        self.logger.info("Tesseract OCR processor [cuda={}]".format(cuda))

        if not self.is_available():
            self.logger.error("Tesseract OCR is not available")
            return
        self.models_dir = os.path.abspath(models_dir)
        self.logger.info(f"Tesseract data dir : {models_dir}")

    def is_available(self) -> bool:
        try:
            import pytesseract

            return True
        except ImportError:
            return False

    def recognize_from_fragments(
        self, images, **kwargs
    ) -> typing.List[typing.Dict[str, any]]:
        """Recognize text from image fragments

        Args:
            images: A list of input images, supplied as numpy arrays with shape
                (H, W, 3).
        """

        print("ICR processing : recognize_from_boxes via boxes")
        if not self.is_available():
            raise Exception("Tesseract OCR is not available, please install it")

        from pytesseract import image_to_data

        results = []
        config = f"--psm 8 --tessdata-dir {self.models_dir}"

        for i, image in enumerate(images):
            try:
                data = image_to_data(
                    image, config=config, lang="eng", output_type="dict"
                )
                n_boxes = len(data['text'])

                words = [
                    {
                        'text': data['text'][i],
                        'conf': data['conf'][i],
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'right': data['left'][i] + data['width'][i],
                        'bottom': data['top'][i] + data['height'][i],
                    }
                    for i in range(n_boxes)
                    if data['text'][i]
                ]

                text = " ".join([w['text'] for w in words])
                text = text.upper() if text is not None else ""
                confidence = (
                    sum([w['conf'] for w in words]) / len(words)
                    if len(words) > 0
                    else 0
                )
                # scale confidence to 0-1
                confidence = round(confidence / 100.0, 4)
                results.append({"confidence": confidence, "text": text, "id": i})
            except Exception as ex:
                self.logger.warning(f"Failed to process image : {ex}")
                results.append({"text": "", "confidence": 0, "id": i})

        return results
