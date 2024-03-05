import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from transformers import (
    AutoModel,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...embeddings.transformers.transformers_embeddings import TransformersEmbeddings
from .base import BaseTemplateMatcher
from .vqnnf.matching.feature_extraction import PixelFeatureExtractor
from .vqnnf.matching.sim import crop_to_content, resize_image, similarity_score
from .vqnnf.matching.template_matching import VQNNFMatcher


def get_model_info_clip(model_ID, device):
    model = CLIPModel.from_pretrained(model_ID).to(device)
    processor = CLIPProcessor.from_pretrained(model_ID)
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)

    return model, processor, tokenizer


def get_model_info(model_id, device):
    """prepare for the model"""
    # Method:2 Create Layout processor with custom future extractor
    # Max model size is 512, so we will need to handle any documents larger than that
    pretrained_model_name_or_path = "microsoft/layoutlmv3-base"
    model = AutoModel.from_pretrained(pretrained_model_name_or_path).to(device)
    feature_extractor = LayoutLMv3FeatureExtractor(
        apply_ocr=False, do_resize=True, resample=Image.BILINEAR
    )
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(pretrained_model_name_or_path)
    processor = LayoutLMv3Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return model, processor, tokenizer


def get_single_text_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    text_embeddings = model.get_text_features(**inputs)
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np


def get_single_image_embedding_clip(
    model, processor, image, words, boxes, device
):  ## resize the image to 224x224

    return get_single_text_embedding(model, processor, text=" ".join(words))
    # image = image.resize((224, 224))
    image = processor(text=None, images=image, return_tensors="pt")["pixel_values"].to(
        device
    )

    embedding = model.get_image_features(image)
    # convert the embeddings to numpy array
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np


def get_single_image_embedding(model, processor, image, words, boxes, device):
    ## resize the image to 224x224
    # image = image.resize((224, 224))
    encoding = processor(
        # fmt: off
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=512,
        # fmt: on
    ).to(model.device)

    # get embeddings from the model
    with torch.no_grad():
        model_output = model(**encoding)
        # get the last_hidden_state from the model_output
        image_features = model_output.last_hidden_state
        # get the mean of the features
        image_features = image_features.mean(dim=1)
        # convert the embeddings to numpy array
        image_features_as_np = image_features.cpu().detach().numpy()

        return image_features_as_np


def augment_document(glow_radius, glow_strength, src_image):
    if True:
        return src_image
        img_blurred = cv2.GaussianBlur(src_image, (glow_radius, glow_radius), 1)
        return img_blurred

    # dilate and erode to get the glow
    # img_blurred = cv2.erode(src_image, np.ones((3, 3), np.uint8), iterations=1)
    img_dilated = cv2.dilate(src_image, np.ones((3, 3), np.uint8), iterations=2)

    # change the color of dilated image
    img_dilated[:, :, 0] = 255
    overlay = cv2.addWeighted(img_dilated, 0.4, src_image, 1, 1).astype(np.uint8)
    return overlay

    # img_dilated = cv2.GaussianBlur(img_dilated, (glow_radius, glow_radius), 1)
    # img_dilated = img_dilated.astype(np.uint8)
    # img_dilated = cv2.addWeighted(src_image, 1, img_dilated, 1, 0)

    max_val = np.max(img_blurred, axis=2)
    # max_val[max_val < 160] = 160
    # max_val[max_val > 200] = 255
    max_val = max_val.astype(np.uint8)
    max_val = np.stack(
        [max_val, np.zeros_like(max_val), np.zeros_like(max_val)], axis=2
    )

    max_val = cv2.GaussianBlur(max_val, (glow_radius, glow_radius), 1)
    return max_val


def odd(f):
    return int(np.ceil(f)) // 2 * 2 + 1


model_ID = "openai/clip-vit-base-patch32"  # "openai/clip-vit-base-patch32"
# Get model, processor & tokenizer
model, processor, tokenizer = get_model_info(model_ID, device="cuda")

embeddings_processor = TransformersEmbeddings(
    model_name_or_path="hf://microsoft/layoutlmv3-base"
)


def get_embedding_feaature(image: Image, words: list, boxes: list) -> np.ndarray:
    embedding = embeddings_processor.get_embeddings(
        texts=words, boxes=boxes, image=Image.fromarray(image)
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
        super().__init__(**kwargs)
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

        # self.feature_extractor_sim = self.feature_extractor
        self.feature_extractor_sim = PixelFeatureExtractor(
            model_name=self.model_name, num_features=self.n_feature
        )

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
    ) -> list[dict[str, tuple[int, int, Any, Any] | Any]]:

        feature_extractor = self.feature_extractor

        similarities = []
        temp_ws = []
        temp_hs = []
        image_sizes = []
        temp_match_time = []
        kmeans_time = []
        xs = []
        ys = []
        ws = []
        hs = []
        predictions = []
        score_threshold = 0.60

        for idx, (template_raw, template_bbox, template_label) in enumerate(
            zip(template_frames, template_boxes, template_labels)
        ):
            print("template_label", template_label)
            print("template_bbox", template_bbox)

            x, y, w, h = [int(t) for t in template_bbox]

            template_plot = cv2.rectangle(
                template_raw.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            template_image = template_raw.copy()
            query_image = frame.copy()

            template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
            query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

            glow_strength = 1  # 0: no glow, no maximum
            glow_radius = 25  # blur radius

            # Only modify the RED channel
            if glow_strength > 0:
                template_image = cv2.cvtColor(template_image, cv2.COLOR_RGB2BGR)
                query_image = cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR)

                template_image = augment_document(
                    glow_radius, glow_strength, template_image
                )
                query_image = augment_document(glow_radius, glow_strength, query_image)

                # cv2.imwrite(f"{exp_folder}/{idx + 1}_overlay_template_GLOW.png", template_image)
                # cv2.imwrite(f"{exp_folder}/{idx + 1}_overlay_query_GLOW.png", query_image)

                # expect RGB images
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

            if True:  # verbose:
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

            predictions.append(
                {
                    "bbox": image_pd,
                    "label": template_label,
                    "score": round(sim_val, 3),
                    "similarity": round(sim_val, 3),
                }
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
        t = cv2.resize(template_snippet, (224, 224), interpolation=cv2.INTER_AREA)
        q = cv2.resize(query_pred_snippet, (224, 224), interpolation=cv2.INTER_AREA)

        template_snippet_features = self.feature_extractor_sim.get_features(t)
        query_pred_snippet_features = self.feature_extractor_sim.get_features(q)

        cosine = nn.CosineSimilarity(dim=1)
        feature_sim = cosine(
            template_snippet_features.reshape(1, -1),
            query_pred_snippet_features.reshape(1, -1),
        )

        if True:
            words = ["claim", "provider"]
            boxes = [[0, 0, 100, 100], [100, 100, 200, 200]]

            template_snippet_features = get_embedding_feaature(t, words, boxes)
            query_pred_snippet_features = get_embedding_feaature(
                q,
                words=["claim"],
                boxes=[[0, 0, 100, 100]],
            )

            cosine = nn.CosineSimilarity(dim=1)
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


def mask_iou(mask1, mask2):
    # convert to binary
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    # save the masks
    return iou_score
