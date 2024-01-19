import os
import time
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from skimage.exposure import rescale_intensity
from torch import nn

from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from .base import BaseTemplateMatcher
from .vqnnf.matching.feature_extraction import PixelFeatureExtractor
from .vqnnf.matching.sim import similarity_score, similarity_score_color
from .vqnnf.matching.template_matching import VQNNFMatcher


def augment_document(glow_radius, glow_strength, src_image):
    if True:
        img_blurred = cv2.GaussianBlur(src_image, (glow_radius, glow_radius), 1)
        # return img_blurred

    max_val = np.max(img_blurred, axis=2)
    # max_val[max_val < 160] = 160
    # max_val[max_val > 200] = 255
    max_val = max_val.astype(np.uint8)
    max_val = np.stack(
        [max_val, np.zeros_like(max_val), np.zeros_like(max_val)], axis=2
    )
    max_val = cv2.GaussianBlur(max_val, (glow_radius, glow_radius), 1)
    # combine the two images
    # img_blended = cv2.addWeighted(src_image, 1, max_val, .8, 0)

    return max_val


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
        # model,n_features,n_codes,rect_haar_filters,scale,pca_dims,M_IOU,Success_Rate,Temp_Match_Time,Kmeans_Time,Total_Time
        # efficientnet-b0,512.0,128.0,1.0,3.0,,0.833204448223114,1.0,0.033192422654893666,0.03436501820882162,0.06755744086371529
        self.model_name = "efficientnet-b0"
        self.n_feature = 512
        self.n_code = 128
        self.rect_haar_filter = 1
        self.scale = 3
        self.pca_dim = None
        self.feature_extractor = PixelFeatureExtractor(
            model_name=self.model_name, num_features=self.n_feature
        )

    def load(self, model_name_or_path: Union[str, os.PathLike]) -> nn.Module:
        pass

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
            sim_val = similarity_score(template_snippet, query_pred_snippet, "ssim")
            similarities.append(sim_val)

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

            # extract heatmaps for template and query
            if True:
                query_pred_nnf = query_nnf[qys : qys + qhs, qxs : qxs + qws, :] * 255
                template_nnf = template_nnf * 255

                template_nnf = template_nnf.astype(np.uint8)
                query_pred_nnf = query_pred_nnf.astype(np.uint8)

                # query_pred_nnf = rescale_intensity(query_pred_nnf, out_range=(0, 255))
                # template_nnf = rescale_intensity(template_nnf, out_range=(0, 255))

                # cv2.imwrite(f"/tmp/dim/{idx}_query_hmap_nnf.png", query_pred_nnf)
                # cv2.imwrite(f"/tmp/dim/{idx}_template_nnf.png", template_nnf)

                # sim_val = similarity_score_color(template_nnf, query_pred_nnf, "ssim")

                heatmap = rescale_intensity(heatmap, out_range=(0, 255))
                #
                # heatmap = cv2.applyColorMap(
                #     (
                #         ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))
                #         * 255
                #     ).astype(np.uint8),
                #     cv2.COLORMAP_JET,
                # )

                hmap_snippet = heatmap[qys : qys + qhs, qxs : qxs + qws]

                hmap_intensity = np.sum(hmap_snippet) / (
                    hmap_snippet.shape[0] * hmap_snippet.shape[1] * 255
                )

                # compute the intensity of the maximum value in the heatmap
                # ptx, pty = np.unravel_index(np.argmax(hmap_snippet), hmap_snippet.shape)
                # val = hmap_snippet[ptx, pty]
                #
                # cv2.imwrite(
                #     f"/tmp/dim/{idx}_query_hmap_{hmap_intensity}.png", hmap_snippet
                # )

            predictions.append(
                {
                    "bbox": image_pd,
                    "label": template_label,
                    "score": round(sim_val, 3),
                    "similarity": round(sim_val, 3),
                }
            )

            cv2.imwrite(f"/tmp/dim/{idx}_query_pd_snippet.png", query_pred_snippet)
            cv2.imwrite(f"/tmp/dim/{idx}_template_snippet.png", template_snippet)

            break

        return predictions
