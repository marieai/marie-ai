import os
from typing import List, Optional, Union

import cv2
import numpy as np
import torch

from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...utils.overlap import merge_bboxes_as_block
from ...utils.utils import ensure_exists
from .base import BaseTemplateMatcher
from .model import TemplateMatchResult


class MetaTemplateMatcher(BaseTemplateMatcher):
    """
    This class is used to match a template in an image using the Pattern Matching algorithm.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        labels: Optional[List[str]] = None,
        batch_size: int = 16,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        """

        :param model_name_or_path: Directory of a saved model or the name of a public model  from the HuggingFace model hub.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param use_gpu: Whether to use GPU (if available).
        :param batch_size: Number of Documents to be processed at a time.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__(False, **kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Document matcher : {model_name_or_path}")
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
        self.labels = labels
        self.progress_bar = False

        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )

        self.device = resolved_devices[0]

    def predict(
        self,
        frame: np.ndarray,
        template_frames: list[np.ndarray],
        template_boxes: list[tuple[int, int, int, int]],
        template_labels: list[str],
        template_texts: list[str] = None,
        score_threshold: float = 0.9,
        scoring_strategy: str = "EXACT",
        batch_size: int = 1,
        words: list[str] = None,
        word_boxes: list[tuple[int, int, int, int]] = None,
        word_lines: list[tuple[int, int, int, int]] = None,
    ) -> list[TemplateMatchResult]:

        predictions = []
        if words is None or word_boxes is None:
            self.logger.warning("No words or word_boxes provided. Skipping prediction.")
            return predictions

        assert len(words) == len(word_boxes)
        assert len(template_frames) == len(template_boxes)
        assert len(template_texts) == len(template_labels)
        # extract type (EXACT, REGEX, FUZZY, EMBEDDINGS) from the template_labels

        page_words = words
        page_boxes = word_boxes
        word_lines = word_lines
        k = 0

        for idx, (template_text, template_label) in enumerate(
            zip(template_texts, template_labels)
        ):
            ngram = len(template_text.split(" "))
            ngrams = [ngram - 1, ngram, ngram + 1]
            ngrams = [n for n in ngrams if 0 < n <= len(page_words)]

            for ngram in ngrams:
                for i in range(len(page_words) - ngram + 1):
                    ngram_words = page_words[i : i + ngram]
                    ngram_boxes = page_boxes[i : i + ngram]
                    if word_lines:
                        ngram_lines = word_lines[i : i + ngram]
                        if len(set(ngram_lines)) > 1:
                            self.logger.debug(
                                f"Skipping ngram {ngram_words} as it is not in the same line"
                            )
                            continue

                    key = "_".join(ngram_words)
                    box = merge_bboxes_as_block(ngram_boxes)
                    x, y, w, h = box
                    snippet = frame[y : y + h, x : x + w :]
                    ngram_words = " ".join(ngram_words).strip().upper()
                    template_text = template_text.strip().upper()
                    sim_val = self.score(ngram_words, template_text, snippet)

                    if ngram_words == template_text or sim_val > score_threshold:
                        predictions.append(
                            TemplateMatchResult(
                                bbox=box,
                                label=template_label,
                                score=sim_val,
                                similarity=sim_val,
                            )
                        )

                        ensure_exists(f"/tmp/fragments/converted/{key}")
                        cv2.imwrite(
                            f"/tmp/fragments/meta/{k}_{round(sim_val, 3)}.png", snippet
                        )
                        k += 1

        return predictions

    def score(self, ngram_words: str, template_text: str, query_pred_snippet) -> float:
        from Levenshtein import distance

        # resize the images to be the same size
        # q = resize_image(query_pred_snippet, (72, 224))[0]

        d = distance(ngram_words, template_text)
        sim_val = 1 - d / max(len(ngram_words), len(template_text))

        # TODO : Implement embeddings similarity

        return sim_val
