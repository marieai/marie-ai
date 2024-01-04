from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from docarray import DocList

from marie.api.docs import MarieDoc
from marie.logging.logger import MarieLogger


class BaseTemplateMatcher(ABC):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        self.logger = MarieLogger(self.__class__.__name__).logger

    @abstractmethod
    def predict(
        self,
        documents: DocList[MarieDoc],
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocList[MarieDoc]:
        pass

    def run(
        self,
        documents: DocList[MarieDoc],
        templates: List[np.ndarray],
        threshold: float = 0.9,
        regions: List[int, int, int, int] = None,
        batch_size: Optional[int] = None,
    ) -> DocList[MarieDoc]:
        """
        Run the template matching on the given documents.

        :param threshold:
        :param documents: List of documents to run template matching on
        :param templates:
        :param regions:
        :param batch_size:
        :return:
        """

        raise NotImplementedError
