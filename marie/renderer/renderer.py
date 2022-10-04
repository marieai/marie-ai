from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Dict, Union

import numpy as np


class ResultRenderer(ABC):
    def __init__(self, config={}):
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
        frames: [np.array],
        results: [Dict[str, Any]],
        output_filename: Union[str, PathLike],
    ) -> None:
        """
        Result renderer that renders results to output

        Args:
            frames ([np.array]): A URI supported by this PathHandler
            results ([[Dict[str, Any]]): A OCR results array
            output_filename (Union[str, PathLike]): a file path which exists on the local file system
        Returns:
            None
        """
        pass
