from typing import List, Optional, Union

import numpy as np
import sahi.annotation as BoundingBox
from sahi.prediction import ObjectPrediction, PredictionScore

from marie.logging.logger import MarieLogger

from ...logging.profile import TimeContext
from .base import BaseTemplateMatcher
from .model import TemplateMatchResult


class CompositeTemplateMatcher(BaseTemplateMatcher):
    """
    CompositeTemplateMatcher is used to match a template in an image using multiple algorithms.
    """

    def __init__(
        self,
        matchers: List[BaseTemplateMatcher],
        break_on_match: bool = False,
        show_error: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        super().__init__(False, **kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.show_error = show_error  # show prediction errors
        self.progress_bar = False
        self.matchers = matchers
        self.break_on_match = break_on_match

    def predict(
        self,
        frame: np.ndarray,
        template_frames: list[np.ndarray],
        template_boxes: list[tuple[int, int, int, int]],
        template_labels: list[str],
        template_texts: list[str] = None,
        score_threshold: float = 0.9,
        scoring_strategy: str = "weighted",
        max_objects: int = 1,
        batch_size: int = 1,
        words: list[str] = None,
        word_boxes: list[tuple[int, int, int, int]] = None,
        word_lines: list[tuple[int, int, int, int]] = None,
    ) -> list[TemplateMatchResult]:
        raise NotImplementedError(
            "This method is not implemented in CompositeTemplateMatcher"
        )

    def run(
        self,
        frames: list[np.ndarray],
        template_frames: list[np.ndarray],
        template_boxes: list[tuple[int, int, int, int]],
        template_labels: list[str],
        template_texts: list[str] = None,
        metadata: Optional[Union[dict, list]] = None,
        score_threshold: float = 0.8,
        scoring_strategy: str = "weighted",  # "weighted" or "average"
        max_overlap: float = 0.5,
        max_objects: int = 1,
        window_size: tuple[int, int] = (384, 128),  # h, w
        regions: list[tuple[int, int, int, int]] = None,
        downscale_factor: float = 1.0,
        batch_size: Optional[int] = None,
    ) -> list[TemplateMatchResult]:
        results = []
        postprocess = self.setup_postprocess()

        for matcher in self.matchers:
            with TimeContext(
                f"Evaluating matcher : {matcher.__class__.__name__}", logger=self.logger
            ):
                result = matcher.run(
                    frames=frames,
                    template_frames=template_frames,
                    template_boxes=template_boxes,
                    template_labels=template_labels,
                    template_texts=template_texts,
                    metadata=metadata,
                    score_threshold=score_threshold,
                    scoring_strategy=scoring_strategy,
                    max_overlap=max_overlap,
                    max_objects=max_objects,
                    window_size=window_size,
                    regions=regions,
                    downscale_factor=downscale_factor,
                    batch_size=batch_size,
                )
                results.extend(result)
                if self.break_on_match and result:
                    break

        converted_results = []
        results_by_page = {}
        for result in results:
            if result.frame_index not in results_by_page:
                results_by_page[result.frame_index] = []
            results_by_page[result.frame_index].append(result)

        for page_index, results in results_by_page.items():
            result_bboxes = [result.bbox for result in results]
            result_labels = [result.label for result in results]
            result_scores = [result.score for result in results]
            object_prediction_list: List[ObjectPrediction] = (
                self.to_object_prediction_list(
                    result_bboxes, result_labels, result_scores
                )
            )

            if postprocess is not None:
                object_prediction_list = postprocess(object_prediction_list)
                # convert back to TemplateMatchResult
                for object_prediction in object_prediction_list:
                    bbox: BoundingBox = object_prediction.bbox
                    bbox = bbox.to_xywh()

                    score: PredictionScore = object_prediction.score
                    score = score.value
                    label = object_prediction.category.name

                    converted_results.append(
                        TemplateMatchResult(
                            bbox=bbox,
                            label=label,
                            score=score,
                            similarity=score,
                            frame_index=page_index,
                        )
                    )

        return converted_results
