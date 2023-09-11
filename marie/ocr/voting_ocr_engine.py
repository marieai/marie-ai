from typing import Union, List

import numpy as np
from PIL import Image

from marie.boxes import PSMode
from marie.ocr import OcrEngine, CoordinateFormat


class VotingOcrEngine(OcrEngine):
    """
    An implementation of OcrEngine which includes voting. This is a base class for all voting OCR engines.
    Notes :
    https://www.atalasoft.com/docs/dotimage/docs/html/N_Atalasoft_Ocr_Voting.htm
    https://github.com/HasithaSuneth/Py-Tess-OCR/blob/main/Py-Tess-OCR%20(Linux).py
    """

    def extract(
        self,
        frames: Union[np.ndarray, List[np.ndarray], List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYXY,
        regions: [] = None,
        queue_id: str = None,
        **kwargs
    ):
        raise NotImplementedError
