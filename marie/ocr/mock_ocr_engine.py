import os
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from marie.boxes import PSMode
from marie.boxes.box_processor import BoxProcessor
from marie.common.file_io import get_cache_dir
from marie.constants import __model_path__
from marie.logging_core.logger import MarieLogger
from marie.ocr import CoordinateFormat, OcrEngine
from marie.utils.image_utils import hash_frames_fast
from marie.utils.json import load_json_file
from marie.utils.utils import ensure_exists


class MockOcrEngine(OcrEngine):
    """
    Mock OCR engine that can be used for testing purposes.
    """

    def __init__(
        self,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        *,
        box_processor: Optional[BoxProcessor] = None,
        **kwargs,
    ) -> None:
        super().__init__(cuda=cuda, **kwargs)
        self.logger = MarieLogger(context=self.__class__.__name__)
        work_dir_boxes = ensure_exists("/tmp/boxes")
        work_dir_icr = ensure_exists("/tmp/icr")

    def extract(
        self,
        frames: Union[np.ndarray, List[np.ndarray], List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYWH,
        regions: [] = None,
        queue_id: str = None,
        **kwargs,
    ):

        # create local asset directory
        frame_checksum = hash_frames_fast(frames=frames)
        cache_dir = get_cache_dir()
        generators_dir = os.path.join(cache_dir, "generators")

        root_asset_dir = ensure_exists(os.path.join(generators_dir, frame_checksum))
        json_path = os.path.join(root_asset_dir, "results", "results.json")
        try:
            return load_json_file(json_path)
        except BaseException as error:
            self.logger.error(f"Extract error {error}", exc_info=False)
            raise error
