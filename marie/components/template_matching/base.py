from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

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
        frame: np.ndarray,
        templates: List[np.ndarray],
        labels: List[str],
        score_threshold: float = 0.9,
        max_overlap: float = 0.5,
        max_objects: int = 1,
        region: tuple[int, int, int, int] = None,
        downscale_factor: int = 1,
        batch_size: Optional[int] = None,
    ) -> list[tuple[int, int, int, int]]:
        """
        Find all possible templates locations above a score-threshold, provided a list of templates to search and an image.
        Resulting detections are not filtered by NMS and thus might overlap.Use :meth:`~:run` to perform the search with NMS.
        """
        ...

    def run(
        self,
        frames: list[np.ndarray],
        templates: list[np.ndarray],
        labels: list[str],
        score_threshold: float = 0.9,
        max_overlap: float = 0.5,
        max_objects: int = 1,
        regions: list[tuple[int, int, int, int]] = None,
        downscale_factor: int = 1,
        batch_size: Optional[int] = None,
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        """
        Search each template in the images, and return the best `max_objects` locations which offer the best score and which do not overlap.

        :param frames: A list of images in which to perform the search, it should be the same depth and number of channels that of the templates.
        :param templates: A list of templates as numpy array to search in each image.
        :param labels: A list of labels for each template. The length of this list should be the same as the length of the templates list.
        :param score_threshold: The minimum score to consider a match.
        :param max_overlap: The maximum overlap to consider a match. This is the maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes.
        :param max_objects: The maximum number of objects to return.
        :param regions: A list of regions of interest in the images in the format (x, y, width, height). If None, the whole image is considered.
        :param downscale_factor: The factor by which to downscale the images before performing the search. This is useful to speed up the search.
        :param batch_size: The batch size to use for the prediction.
        :return: A dictionary of lists of bounding boxes in the format (x, y, width, height) for each label per frame.
        """

        # assertions can be disabled via the the -O flag  (python -O)
        assert len(templates) == len(labels)
        assert 0 <= score_threshold <= 1
        assert 0 <= max_overlap <= 1
        assert max_objects > 0
        assert downscale_factor > 0
        assert batch_size is None or batch_size > 0

        if regions is None:
            regions = [(0, 0, image.shape[1], image.shape[0]) for image in frames]

        assert len(frames) == len(regions)

        results = {}

        for i, (frame, region) in enumerate(zip(frames, regions)):
            self.logger.info(f"matching frame {i} region: {region}")

            # check depth and number of channels
            assert frame.ndim == 3
            assert frame.shape[2] == templates[0].shape[2]

            predictions = self.predict(
                frame,
                templates,
                labels,
                score_threshold,
                max_overlap,
                max_objects,
                region,
                downscale_factor,
                batch_size,
            )
            self.logger.info(f"predictions: {predictions}")

            results[i] = {
                "page": i,
                "boxes": predictions,
            }

        return results
