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


def odd(f):
    return int(np.ceil(f)) // 2 * 2 + 1


embeddings_processor = OpenAIEmbeddings(
    # model_name_or_path="hf://openai/clip-vit-large-patch14"
    model_name_or_path="marie/clip-vit-base-patch32"
    # model_name_or_path="hf://openai/clip-vit-base-patch32"
)


def get_embedding_feature(image: np.ndarray, words: list, boxes: list) -> np.ndarray:
    # This is a pre-processing step to get the embeddings for the words and boxes in the image
    print("embedding image", image.shape)
    # image1 = resize_image(image, (72, 224))[0]
    image1 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    embedding = embeddings_processor.get_embeddings(
        texts=words, boxes=boxes, image=image1
    )

    return embedding.embeddings


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
        self.model_name = "efficientnet-b0"
        self.n_feature = 512
        self.n_code = 128
        self.rect_haar_filter = 1
        self.scale = 3
        self.pca_dim = 128
        self.feature_extractor = PixelFeatureExtractor(
            model_name=self.model_name, num_features=self.n_feature
        )

        self.feature_extractor_sim = PixelFeatureExtractor(
            model_name=self.model_name, num_features=512
        )

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

        feature_extractor = self.feature_extractor

        xs = []
        ys = []
        ws = []
        hs = []

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

            template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
            query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

            cv2.imwrite("/tmp/dim/template_plot.png", template_plot)
            cv2.imwrite("/tmp/dim/query_image.png", query_image)
            cv2.imwrite("/tmp/dim/template_image.png", template_image)

            template_image_features = feature_extractor.get_features(template_image)

            temp_x, temp_y, temp_w, temp_h = template_bbox
            temp_x = int(max(temp_x, 0))
            temp_y = int(max(temp_y, 0))
            template_features = template_image_features[
                :, temp_y : temp_y + temp_h, temp_x : temp_x + temp_w
            ]

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

            query_image_features = feature_extractor.get_features(query_image)

            torch.cuda.synchronize()
            t1 = time.time()
            (
                heatmap,
                filt_heatmaps,
                template_nnf,
                query_nnf,
            ) = template_matcher.get_heatmap(query_image_features)
            torch.cuda.synchronize()
            t2 = time.time()

            query_w, query_h = template_bbox[3], template_bbox[2]

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
            sim_val = self.score(template_snippet, query_pred_snippet)

            if sim_val < score_threshold:
                continue

            predictions.append(
                {
                    "bbox": image_pd,
                    "label": template_label,
                    "score": round(sim_val, 3),
                    "similarity": round(sim_val, 3),
                }
            )

            if False:  # verbose:
                cv2.imwrite(
                    f"/tmp/dim/{idx}_template_nnf.png",
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
                    f"/tmp/dim/{idx}_query_nnf.png",
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

            if False:
                cv2.imwrite(
                    f"/tmp/dim/{idx}_query_pd_snippet_{round(sim_val, 3)}.png",
                    query_pred_snippet,
                )
                cv2.imwrite(
                    f"/tmp/dim/{idx}_template_snippet_{round(sim_val, 3)}.png",
                    template_snippet,
                )

            if True:
                stacked = np.hstack((template_snippet, query_pred_snippet))
                cv2.imwrite(
                    f"/tmp/dim/final/stacked_{idx}_{round(sim_val, 3)}.png",
                    stacked,
                )

            break

        return predictions

    def score(self, template_snippet, query_pred_snippet) -> float:

        print("template_snippet", template_snippet.shape)
        print("query_pred_snippet", query_pred_snippet.shape)

        # resize the images to be the same size
        t = resize_image(template_snippet, (72, 224))[0]
        q = resize_image(query_pred_snippet, (72, 224))[0]

        # t = cv2.resize(template_snippet, (224, 224), interpolation=cv2.INTER_AREA)
        # q = cv2.resize(query_pred_snippet, (224, 224), interpolation=cv2.INTER_AREA)

        template_snippet_features = self.feature_extractor_sim.get_features(t)
        query_pred_snippet_features = self.feature_extractor_sim.get_features(q)

        cosine = nn.CosineSimilarity(dim=1)
        feature_sim = cosine(
            template_snippet_features.reshape(1, -1),
            query_pred_snippet_features.reshape(1, -1),
        )

        # TODO : add the embedding similarity for the text
        words = ["word-a", "word-1"]
        boxes = [[0, 0, 100, 100], [100, 100, 200, 200]]

        template_snippet_features = get_embedding_feature(t, words=[], boxes=[])
        query_pred_snippet_features = get_embedding_feature(q, words=[], boxes=[])

        embedding_sim = cosine(
            torch.from_numpy(template_snippet_features.reshape(1, -1)),
            torch.from_numpy(query_pred_snippet_features.reshape(1, -1)),
        )

        embedding_sim = embedding_sim.cpu().numpy()[0]
        feature_sim = feature_sim.cpu().numpy()[0]
        # mask_val = mask_iou(t, q)
        print("sim query/template", feature_sim, embedding_sim)
        sim_val = (feature_sim + embedding_sim) / 2

        return sim_val
