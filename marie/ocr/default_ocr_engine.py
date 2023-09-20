import os
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image

from marie.boxes import PSMode
from marie.constants import __model_path__
from marie.document import TrOcrIcrProcessor
from marie.ocr import OcrEngine, CoordinateFormat


class DefaultOcrEngine(OcrEngine):
    """
    Recognizes text in an image.
    This implementation will select best available OcrEngine based on available models and configs
    Text extraction can either be executed out over the entire image or over selected regions of interests (ROIs)
    aka bounding boxes.
    """

    def __init__(
        self,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(models_dir=models_dir, cuda=cuda, **kwargs)
        self.icr_processor = TrOcrIcrProcessor(
            work_dir=self.work_dir_icr, cuda=self.has_cuda
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
        try:
            return self.process_single(
                self.box_processor,
                self.icr_processor,
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
