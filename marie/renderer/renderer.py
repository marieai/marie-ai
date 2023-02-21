from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Dict, Union, Optional, Callable

import numpy as np

from marie.logging.logger import MarieLogger


class ResultRenderer(ABC):
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        self.logger = MarieLogger(ResultRenderer.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the renderer
        """
        pass

    @abstractmethod
    def render(
        self,
        frames: np.ndarray,
        results: [Dict[str, Any]],
        output_file_or_dir: Union[str, PathLike],
        filename_generator: Optional[Callable[[int], str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Result renderer that renders results to output
        :param frames: A list of frames to render
        :param results: A OCR results array
        :param output_file_or_dir: The output file or directory to render to
        :param filename_generator: a function that generates a filename for each page
        :param kwargs: additional arguments
        :return:
        """

        pass

    def check_format_xywh(self, result, convert=True):
        """
        Check if the page result is in XYWH format
        """
        # Ensure page is in xywh format
        # change from xywy -> xyxy
        meta = result["meta"]
        if convert and meta["format"] != "xywh":
            self.logger.info("Changing coordinate format from xywh -> xyxy")
            for word in result["words"]:
                x, y, w, h = word["box"]
                w_box = [x, y, x + w, y + h]
                word["box"] = w_box
                # FIXME:  BLOWS memory on GPU
                # word["box"] = CoordinateFormat.convert(
                #     word["box"], CoordinateFormat.XYWH, CoordinateFormat.XYXY
                # )
