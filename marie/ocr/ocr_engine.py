import os
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
from PIL import Image

from marie.constants import __model_path__

from marie.boxes import PSMode
from marie.ocr.coordinate_format import CoordinateFormat


class OcrEngine(ABC):
    """
    Recognizes text in an image.
    Text extraction can either be executed out over the entire image or over selected regions of interests (ROIs)
    aka bounding boxes.
    """

    def __init__(
        self,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    @abstractmethod
    def extract(
        self,
        frames: Union[np.ndarray, List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYXY,
        regions: [] = None,
        queue_id: str = None,
        **kwargs,
    ):
        ...
