import os
import traceback
from typing import List, Union

import numpy as np
from PIL import Image

from marie.boxes import PSMode
from marie.constants import __model_path__
from marie.document import TrOcrIcrProcessor
from marie.document.craft_icr_processor import CraftIcrProcessor
from marie.document.tesseract_icr_processor import TesseractOcrProcessor
from marie.ocr import OcrEngine, CoordinateFormat


class VotingOcrEngine(OcrEngine):
    """
    An implementation of OcrEngine which includes voting. This is a base class for all voting OCR engines.
    Notes :
    https://www.atalasoft.com/docs/dotimage/docs/html/N_Atalasoft_Ocr_Voting.htm
    https://github.com/HasithaSuneth/Py-Tess-OCR/blob/main/Py-Tess-OCR%20(Linux).py
    """

    def __init__(
        self,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(models_dir=models_dir, cuda=cuda, **kwargs)

        self.processors = {
            "trocr",
            {
                "enabled": True,
                "processor": TrOcrIcrProcessor(
                    work_dir=self.work_dir_icr, cuda=self.has_cuda
                ),
            },
            "craft",
            {
                "enabled": True,
                "processor": CraftIcrProcessor(
                    work_dir=self.work_dir_icr, cuda=self.has_cuda
                ),
            },
            "tesseract",
            {
                "enabled": True,
                "processor": TesseractOcrProcessor(
                    work_dir=self.work_dir_icr, cuda=self.has_cuda
                ),
            },
        }

    def extract(
        self,
        frames: Union[np.ndarray, List[np.ndarray], List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYXY,
        regions: [] = None,
        queue_id: str = None,
        **kwargs,
    ):

        for processor in self.processors:
            try:
                self.logger.info(f"Processing with {processor.__class__.__name__}")
            except Exception as e:
                print(e)
                traceback.print_exc()
                continue
