from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Dict, Union

import numpy as np

from marie.logging.logger import MarieLogger


class ResultRenderer(ABC):
    def __init__(self, config={}):
        self.logger = MarieLogger(ResultRenderer.__name__)
        pass

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
        output_filename: Union[str, PathLike],
    ) -> None:
        """
        Result renderer that renders results to output

        Args:
            frames (np.ndarray): A URI supported by this PathHandler
            results ([[Dict[str, Any]]): A OCR results array
            output_filename (Union[str, PathLike]): a file path which exists on the local file system
        Returns:
            None
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
