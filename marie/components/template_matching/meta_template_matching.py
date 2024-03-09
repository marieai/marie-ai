import os
import time
from typing import Any, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...embeddings.openai.openai_embeddings import OpenAIEmbeddings
from ...embeddings.transformers.transformers_embeddings import TransformersEmbeddings
from ...utils.resize_image import resize_image
from .base import BaseTemplateMatcher
from .vqnnf.matching.feature_extraction import PixelFeatureExtractor
from .vqnnf.matching.template_matching import VQNNFMatcher


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
        score_threshold: float = 0.9,
        batch_size: Optional[int] = None,
        words: list[str] = None,
        word_boxes: list[tuple[int, int, int, int]] = None,
    ) -> list[tuple[int, int, int, int]]:

        predictions = []

        for idx, (template_raw, template_bbox, template_label) in enumerate(
            zip(template_frames, template_boxes, template_labels)
        ):
            # print("template_label", template_label)
            # print("template_bbox", template_bbox)
            x, y, w, h = [int(t) for t in template_bbox]

            template_plot = cv2.rectangle(
                template_raw.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            template_image = template_raw.copy()
            query_image = frame.copy()

        return predictions

    def score(self, template_snippet, query_pred_snippet) -> float:

        print("template_snippet", template_snippet.shape)
        print("query_pred_snippet", query_pred_snippet.shape)

        # resize the images to be the same size
        t = resize_image(template_snippet, (72, 224))[0]
        q = resize_image(query_pred_snippet, (72, 224))[0]

        sim_val = 0
        return sim_val
