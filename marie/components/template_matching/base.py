from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
import numpy as np
from patchify import patchify

from marie.logging.logger import MarieLogger
from marie.utils.nms import non_max_suppression_fast
from marie.utils.overlap import find_overlap


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
        template_frames: list[np.ndarray],
        template_boxes: list[tuple[int, int, int, int]],
        template_labels: list[str],
        score_threshold: float = 0.9,
        max_overlap: float = 0.5,
        max_objects: int = 1,
        window_size: tuple[int, int] = (384, 128),  # h, w
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
        template_frames: list[np.ndarray],
        template_boxes: list[tuple[int, int, int, int]],
        template_labels: list[str],
        score_threshold: float = 0.45,
        max_overlap: float = 0.5,
        max_objects: int = 1,
        window_size: tuple[int, int] = (384, 128),  # h, w
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
        # assert len(templates) == len(labels)
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

            # validate the template frames are the same size as the window size
            for template_frame in template_frames:
                assert template_frame.shape[0] == window_size[0]
                assert template_frame.shape[1] == window_size[1]

                if (
                    template_frame.shape[0] != window_size[0]
                    or template_frame.shape[1] != window_size[1]
                ):
                    raise Exception(
                        "Template frame size does not match window size, please resize the template frames to match the window size"
                    )

            # downscale the window size
            window_size = (
                int(window_size[0] // downscale_factor),
                int(window_size[1] // downscale_factor),
            )

            print("window_size : ", window_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            patches = patchify(
                frame,
                window_size,
                step=min(int(window_size[0] // 2), int(window_size[1] // 2)),
            )
            patches = patches.reshape(-1, window_size[0], window_size[1])
            print("patches : ", patches.shape)

            agg_bboxes = []
            agg_labels = []
            agg_scores = []

            for idx, patch in enumerate(patches):
                cv2.imwrite(f"/tmp/dim/patch_{idx}.png", patch)
                if idx != 1:
                    continue

                predictions = self.predict(
                    patch,
                    template_frames,
                    template_boxes,
                    template_labels,
                    score_threshold,
                    max_overlap,
                    max_objects,
                    window_size,
                    region,
                    downscale_factor,
                    batch_size,
                )
                self.logger.info(f"predictions: {predictions}")

                for prediction in predictions:
                    agg_bboxes.append(prediction["bbox"])
                    agg_labels.append(prediction["label"])
                    agg_scores.append(prediction["score"])

                results[i] = {
                    "page": i,
                    "predictions": predictions,
                }

            # filter out low scores
            score_threshold = 0.25
            for idx, score in enumerate(agg_scores):
                if score < score_threshold:
                    agg_bboxes.pop(idx)
                    agg_labels.pop(idx)
                    agg_scores.pop(idx)

            print("bboxes before : ", agg_bboxes)
            # sort by score (descending)
            agg_bboxes = [
                bbox
                for _, bbox in sorted(
                    zip(agg_scores, agg_bboxes), key=lambda pair: pair[0], reverse=True
                )
            ]
            agg_labels = [
                label
                for _, label in sorted(
                    zip(agg_scores, agg_labels), key=lambda pair: pair[0], reverse=True
                )
            ]
            agg_scores = sorted(agg_scores, reverse=True)

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            for bbox, label, score in zip(agg_bboxes, agg_labels, agg_scores):
                print(" -- bbox : ", bbox, score)

                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"{score:.2f}",
                    (bbox[0], bbox[1] + bbox[3] // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            cv2.imwrite(f"/tmp/dim/results_frame_{i}.png", frame)

        return results
