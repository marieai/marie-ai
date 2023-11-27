import os
from typing import Any, Dict, List, Union, Optional

import numpy as np
from PIL import Image

from marie.boxes import PSMode
from marie.boxes.box_processor import BoxProcessor
from marie.constants import __model_path__
from marie.document import TrOcrProcessor
from marie.document.ocr_processor import OcrProcessor
from marie.ocr import OcrEngine, CoordinateFormat


class DefaultOcrEngine(OcrEngine):
    """
    Recognizes text in an image.

    This implementation will select best available OcrEngine based on available models and configs.
    Text extraction can either be executed out over the entire image or over selected regions of interests (ROIs)
    aka bounding boxes.

    Args:
        models_dir (str): Path to the directory containing the OCR models.
        cuda (bool): Whether to use CUDA for GPU acceleration.
        **kwargs: Additional keyword arguments to pass to the parent class.

    Attributes:
        ocr_processor (TrOcrProcessor): OCR processor for text recognition.
    """

    def __init__(
        self,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        *,
        box_processor: Optional[BoxProcessor] = None,
        default_ocr_processor: Optional[OcrProcessor] = None,
        **kwargs,
    ) -> None:
        """
        Initializes a new instance of the DefaultOcrEngine class.

        Args:
            models_dir (str): Path to the directory containing the OCR models.
            cuda (bool): Whether to use CUDA for GPU acceleration.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(
            models_dir=models_dir, cuda=cuda, box_processor=box_processor, **kwargs
        )

        self.ocr_processor = (
            default_ocr_processor
            if default_ocr_processor is not None
            else TrOcrProcessor(
                work_dir=self.work_dir_icr,
                cuda=self.has_cuda,
            )
        )

    def extract(
        self,
        frames: Union[np.ndarray, List[np.ndarray], List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYWH,
        regions: [] = None,
        queue_id: str = None,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Extracts text from one or more images.

        Args:
            frames (Union[np.ndarray, List[np.ndarray], List[Image.Image]]): One or more images to extract text from.
            pms_mode (PSMode): The mode to use for page segmentation.
            coordinate_format (CoordinateFormat): The format to use for bounding box coordinates.
            regions ([]): A list of regions of interest (ROIs) to extract text from.
            queue_id (str): The ID of the queue to use for parallel processing.
            **kwargs: Additional keyword arguments to pass to the parent class.

        Returns:
            A list of dictionaries containing the extracted text and its bounding box coordinates.
        """
        try:
            return self.process_single(
                self.box_processor,
                self.ocr_processor,
                frames,
                pms_mode,
                coordinate_format,
                regions,
                queue_id,
                **kwargs,
            )
        except BaseException as error:
            self.logger.error("Extract error", exc_info=True)
            raise error
