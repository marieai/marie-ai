import copy
import os
from distutils.util import strtobool as strtobool
from typing import Dict, Union, Optional, List

import cv2
from PIL import Image
import numpy as np
from marie.constants import __model_path__

from marie.boxes import BoxProcessorUlimDit, PSMode, BoxProcessorCraft
from marie.document import TrOcrIcrProcessor
from marie.logging.logger import MarieLogger
from marie.ocr import OcrEngine, CoordinateFormat
from marie.utils.base64 import encodeToBase64
from marie.utils.image_utils import hash_bytes
from marie.utils.json import load_json_file
from marie.utils.utils import ensure_exists
from marie.utils.image_utils import hash_file, hash_frames_fast


class MockOcrEngine(OcrEngine):
    """
    Mock OCR engine that can be used for testing
    """

    def __init__(
        self,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.logger = MarieLogger(context=self.__class__.__name__)
        work_dir_boxes = ensure_exists("/tmp/boxes")
        work_dir_icr = ensure_exists("/tmp/icr")

    def extract(
        self,
        frames: Union[np.ndarray, List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYWH,
        regions: [] = None,
        queue_id: str = None,
        **kwargs,
    ):
        try:
            results = load_json_file(
                "/home/gbugaj/tmp/marie-cleaner/169150505/results.json"
            )

            return results
        except BaseException as error:
            self.logger.error("Extract error", error)
            raise error
