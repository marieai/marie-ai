import os
import time
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...embeddings.openai.openai_embeddings import OpenAIEmbeddings
from ...utils.resize_image import resize_image
from .base import BaseTemplateMatcher
from .model import TemplateMatchResult
from .vqnnf.matching.feature_extraction import PixelFeatureExtractor
from .vqnnf.matching.template_matching import VQNNFMatcher


def odd(f):
    return int(np.ceil(f)) // 2 * 2 + 1


class VQNNFTemplateMatcher(BaseTemplateMatcher):
    """
    Efficient High-Resolution Template Matching with Vector Quantized Nearest Neighbour Fields
    https://arxiv.org/pdf/2306.15010.pdf
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
        super().__init__(True, **kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"VQNNF matcher model : {model_name_or_path}")
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
        self.model_name = "efficientnet-b0"
        self.n_feature = 512
        self.n_code = 128
        self.rect_haar_filter = 1
        self.scale = 3
        self.pca_dim = 128

        self.feature_extractor = PixelFeatureExtractor(
            model_name=self.model_name, num_features=self.n_feature
        )
        self.feature_extractor_sim = self.feature_extractor

        self.embeddings_processor = OpenAIEmbeddings(
            model_name_or_path="marie/clip-snippet-rn50x4"
            # model_name_or_path="hf://openai/clip-vit-base-patch32"
        )

        self.cached_features = {}
        self.cached_embeddings_clips = {}

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

        feature_extractor = self.feature_extractor

        xs = []
        ys = []
        ws = []
        hs = []

        predictions = []
        query_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        query_image_features = feature_extractor.get_features(query_image)

        for idx, (template_image, template_bbox, template_label) in enumerate(
            zip(template_frames, template_boxes, template_labels)
        ):
            x, y, w, h = [int(t) for t in template_bbox]
            cache_key = f"key_{x}_{y}_{w}_{h}"
            # template_snippet = template_image[:, y: y + h, x: x + w]
            # cache_key = template_snippet.tobytes()
            template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)

            if cache_key not in self.cached_features:
                self.cached_features[cache_key] = feature_extractor.get_features(
                    template_image
                )
            template_image_features = self.cached_features[cache_key]

            temp_x, temp_y, temp_w, temp_h = template_bbox
            temp_x = int(max(temp_x, 0))
            temp_y = int(max(temp_y, 0))
            template_features = template_image_features[
                :, temp_y : temp_y + temp_h, temp_x : temp_x + temp_w
            ]

            if True:
                template_plot = cv2.rectangle(
                    template_image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2
                )
                fragment = template_image[
                    temp_y : temp_y + temp_h, temp_x : temp_x + temp_w
                ]

                cv2.imwrite("/tmp/dim/template_plot.png", template_plot)
                cv2.imwrite("/tmp/dim/query_image.png", query_image)
                cv2.imwrite("/tmp/dim/template_image.png", template_image)
                cv2.imwrite("/tmp/dim/template_bbox_fragment.png", fragment)

            template_matcher = VQNNFMatcher(
                template=template_features,
                pca_dims=self.pca_dim,
                n_code=self.n_code,
                filters_cat="haar",
                filter_params={
                    "kernel_size": 3,
                    "sigma": 2,
                    "n_scales": self.scale,
                    "filters": self.rect_haar_filter,
                },
                verbose=True,
            )

            (
                heatmap,
                filt_heatmaps,
                template_nnf,
                query_nnf,
            ) = template_matcher.get_heatmap(query_image_features)

            query_w, query_h = template_bbox[3], template_bbox[2]

            for k in range(max_objects):
                query_x, query_y = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                query_x = int(query_x + 1 - (odd(query_w) - 1) / 2)
                query_y = int(query_y + 1 - (odd(query_h) - 1) / 2)

                xs.append(query_y)
                ys.append(query_x)
                ws.append(
                    query_h
                )  # This looks like a bug, but it's not. we have to swap h and w here
                hs.append(query_w)

                # This looks like a bug, but it's not. we have to swap h and w here
                qxs = query_y
                qys = query_x
                qws = query_h
                qhs = query_w

                query_pred_snippet = query_image[
                    qys : qys + qhs, qxs : qxs + qws, :
                ]  # three channels

                template_snippet = template_image[
                    max(temp_y, 0) : min(temp_y + temp_h, template_image.shape[0]),
                    max(temp_x, 0) : min(temp_x + temp_w, template_image.shape[1]),
                    :,
                ]
                image_pd = (qxs, qys, qws, qhs)
                if template_snippet.shape != query_pred_snippet.shape:
                    if template_snippet.shape[0] == 0 or template_snippet.shape[1] == 0:
                        self.logger.warning("Template snippet is empty")
                        continue
                    if (
                        query_pred_snippet.shape[0] == 0
                        or query_pred_snippet.shape[1] == 0
                    ):
                        self.logger.warning("Query snippet is empty")
                        continue
                    query_pred_snippet = cv2.resize(
                        query_pred_snippet,
                        (template_snippet.shape[1], template_snippet.shape[0]),
                    )

                sim_val = self.score(
                    template_snippet, query_pred_snippet, scoring_strategy
                )
                if sim_val < score_threshold:
                    break

                predictions.append(
                    TemplateMatchResult(
                        bbox=image_pd,
                        label=template_label,
                        score=sim_val,
                        similarity=sim_val,
                        frame_index=-1,
                    )
                )

                if False:  # verbose:
                    cv2.imwrite(
                        f"/tmp/dim/{idx}_{k}_template_nnf.png",
                        cv2.applyColorMap(
                            (
                                (
                                    (template_nnf - template_nnf.min())
                                    / (template_nnf.max() - template_nnf.min())
                                )
                                * 255
                            ).astype(np.uint8),
                            cv2.COLORMAP_JET,
                        ),
                    )

                    cv2.imwrite(
                        f"/tmp/dim/{idx}_{k}_query_nnf.png",
                        cv2.applyColorMap(
                            (
                                (
                                    (query_nnf - query_nnf.min())
                                    / (query_nnf.max() - query_nnf.min())
                                )
                                * 255
                            ).astype(np.uint8),
                            cv2.COLORMAP_JET,
                        ),
                    )

                    cv2.imwrite(
                        f"/tmp/dim/{idx}_{k}_heatmap.png",
                        cv2.applyColorMap(
                            (
                                (
                                    (heatmap - heatmap.min())
                                    / (heatmap.max() - heatmap.min())
                                )
                                * 255
                            ).astype(np.uint8),
                            cv2.COLORMAP_JET,
                        ),
                    )

                if False:
                    cv2.imwrite(
                        f"/tmp/dim/{idx}_{k}_query_pd_snippet_{round(sim_val, 3)}.png",
                        query_pred_snippet,
                    )
                    cv2.imwrite(
                        f"/tmp/dim/{idx}_{k}_template_snippet_{round(sim_val, 3)}.png",
                        template_snippet,
                    )

                if True:
                    stacked = np.hstack((template_snippet, query_pred_snippet))
                    cv2.imwrite(
                        f"/tmp/dim/final/stacked_{idx}_{k}__{round(sim_val, 3)}.png",
                        stacked,
                    )

                # set all the values to a low valu on heatmap to avoid duplicates in the next iteration
                heatmap[qys : qys + qhs, qxs : qxs + qws] = -0.82
        return predictions

    def score(
        self, template_snippet, query_pred_snippet, scoring_strategy: str
    ) -> float:
        # resize the images to be the same size
        # 224x224 is the default size for the efficientnet model
        # we use the 224x224 for the feature extraction and the 384x384 for the embedding extraction
        t_clip = resize_image(template_snippet, (224, 224))[0]
        q_clip = resize_image(query_pred_snippet, (224, 224))[0]
        cosine = nn.CosineSimilarity(dim=1)

        t = t_clip
        q = q_clip

        template_snippet_features = self.feature_extractor_sim.get_features(t)
        query_pred_snippet_features = self.feature_extractor_sim.get_features(q)

        feature_sim = cosine(
            template_snippet_features.view(1, -1),
            query_pred_snippet_features.view(1, -1),
        )

        # TODO : add the embedding similarity for the text
        words = []
        boxes = []

        template_snippet_features = self.get_embedding_feature(
            t_clip, words=[], boxes=[]
        )
        query_pred_snippet_features = self.get_embedding_feature(
            q_clip, words=[], boxes=[]
        )

        embedding_sim = cosine(
            torch.from_numpy(template_snippet_features).view(1, -1),
            torch.from_numpy(query_pred_snippet_features).view(1, -1),
        )

        embedding_sim = embedding_sim.cpu().numpy()[0]
        feature_sim = feature_sim.cpu().numpy()[0]
        # feature_sim = 1
        # we already know that the feature similarity is very high for the same image so we can use it as a weight
        if scoring_strategy == "weighted":
            embedding_weight = 0.95
            feature_weight = 0.05
            sim_val = (feature_sim * feature_weight) + (
                embedding_sim * embedding_weight
            )
        elif scoring_strategy == "max":
            sim_val = max(feature_sim, embedding_sim)
        else:
            sim_val = (feature_sim + embedding_sim) / 2
        sim_val = max(0, min(1, sim_val))
        return sim_val

    def get_embedding_feature(
        self, image: np.ndarray, words: list, boxes: list
    ) -> np.ndarray:
        key = image.tobytes()
        if key in self.cached_embeddings_clips:
            return self.cached_embeddings_clips[key]

        # This is a pre-processing step to get the embeddings for the words and boxes in the image
        # this is critical for the similarity calculation that we resize with PADDING and not just scaling
        # image = resize_image(image, (224, 224))[0]
        if image.shape[0] != 224 or image.shape[1] != 224:
            raise ValueError("Image must be 224x224")

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        embedding = self.embeddings_processor.get_embeddings(
            texts=words, boxes=boxes, image=image
        )

        self.cached_embeddings_clips[key] = embedding.embeddings
        return embedding.embeddings
