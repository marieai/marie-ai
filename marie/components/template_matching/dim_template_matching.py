import os
from typing import List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from patchify import patchify
from sewar.full_ref import rmse
from skimage import metrics
from torch import nn
from torchvision import models, transforms

from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...utils.image_utils import crop_to_content
from ...utils.resize_image import resize_image
from .base import BaseTemplateMatcher
from .dim.DIM import odd
from .dim.feature_extractor_vgg import Featex, apply_DIM


class DeepDimTemplateMatcher(BaseTemplateMatcher):
    """
    Robust Template Matching via Hierarchical Convolutional Features from a Shape Biased CNN
    https://arxiv.org/abs/2007.15817
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
        self.model = self.load(model_name_or_path)
        self.layers = (0, 5, 7)  # .85/

        self.featex = Featex(
            self.model,
            True,
            layer1=self.layers[0],
            layer2=self.layers[1],
            layer3=self.layers[2],
        )

    def load(self, model_name_or_path: Union[str, os.PathLike]) -> nn.Module:
        # model = models.vgg19(weights=models.vgg.VGG19_Weights.IMAGENET1K_V1)
        model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
        model = model.features
        model.eval()
        model.to(self.device)

        return model

    def viz_patches(self, patches, filename):
        plt.figure(figsize=(9, 9))
        square_x = patches.shape[1]
        square_y = patches.shape[0]

        ix = 1
        for i in range(square_y):
            for j in range(square_x):
                # specify subplot and turn of axis
                ax = plt.subplot(square_y, square_x, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot
                plt.imshow(patches[i, j, :, :], cmap="gray")
                ix += 1
        # show the figure
        # plt.show()
        plt.savefig(filename)
        plt.close()

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

        featex = self.featex
        layer1, layer2, layer3 = self.layers
        image = frame
        image_plot = image.copy()

        for template_raw, template_bbox, template_label in zip(
            template_frames, template_boxes, template_labels
        ):
            print("template_label", template_label)
            print("template_bbox", template_bbox)

            x, y, w, h = [int(round(t)) for t in template_bbox]
            template_plot = cv2.rectangle(
                template_raw.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            # cv2.imwrite("/tmp/dim/template_plot.png", template_plot)

            image_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            # ensure that the image is in RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # ensure that the template is in RGB
            if len(template_raw.shape) == 2:
                template_raw = cv2.cvtColor(template_raw, cv2.COLOR_GRAY2RGB)

            if image.shape[2] == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if template_raw.shape[2] == 2:
                template_raw = cv2.cvtColor(template_raw, cv2.COLOR_GRAY2RGB)

            T_image = image_transform(template_raw.copy()).unsqueeze(0)
            T_search_image = image_transform(image.copy()).unsqueeze(0)

            # Section 4.2 in the paper

            if 0 <= layer1 <= 4:
                a = 1
            if 4 < layer1 <= 9:
                a = 2
            if 9 < layer1 <= 18:
                a = 4
            if 18 < layer1 <= 27:
                a = 8
            if 27 < layer1 <= 36:
                a = 16
            if 0 <= layer3 <= 4:
                b = 1
            if 4 < layer3 <= 9:
                b = 2
            if 9 < layer3 <= 18:
                b = 4
            if 18 < layer3 <= 27:
                b = 8
            if 27 < layer3 <= 36:
                b = 16

            if w * h <= 4000:
                I_feat = featex(T_image)
                SI_feat = featex(T_search_image)
                resize_bbox = [i / a for i in template_bbox]
            else:
                I_feat = featex(T_image, "big")
                SI_feat = featex(T_search_image, "big")
                resize_bbox = [i / b for i in template_bbox]

            print(" ")
            print("Feature extraction done.")
            pad1 = [int(round(t)) for t in (template_bbox[3], template_bbox[2])]
            pad2 = [int(round(t)) for t in (resize_bbox[3], resize_bbox[2])]

            SI_pad = torch.from_numpy(
                np.pad(
                    SI_feat.cpu().numpy(),
                    ((0, 0), (0, 0), (pad2[0], pad2[0]), (pad2[1], pad2[1])),
                    "symmetric",
                )
            )
            similarity = apply_DIM(I_feat, SI_pad, resize_bbox, pad2, pad1, image, 5)
            print("Matching done.")
            print("similarity shape:", similarity.shape)

            ptx, pty = np.where(similarity == np.amax(similarity))
            image_pd = [
                pty[0] + 1 - (odd(template_bbox[2]) - 1) / 2,
                ptx[0] + 1 - (odd(template_bbox[3]) - 1) / 2,
                template_bbox[2],
                template_bbox[3],
            ]
            image_pd = [int(t) for t in image_pd]
            # clip box to image size
            image_pd[0] = max(image_pd[0], 0)
            image_pd[1] = max(image_pd[1], 0)
            print("Predict box:", image_pd)

            if False:
                # Plotting
                xp, yp, wp, hp = [int(round(t)) for t in image_pd]
                image_plot = cv2.rectangle(
                    image_plot,
                    (int(xp), int(yp)),
                    (int(xp + wp), int(yp + hp)),
                    (255, 0, 0),
                    2,
                )

                fig, ax = plt.subplots(1, 3, figsize=(20, 5))
                plt.ion()
                ax[0].imshow(template_plot)
                ax[1].imshow(image_plot)
                ax[2].imshow(similarity, "jet")

                plt.savefig("/tmp/dim/results.png")
                plt.pause(0.0001)
                plt.close()

            # get template snippet from template image
            template = template_raw[y : y + h, x : x + w, :]
            pred_snippet = image[
                image_pd[1] : image_pd[1] + h, image_pd[0] : image_pd[0] + w, :
            ]

            image1_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(pred_snippet, cv2.COLOR_BGR2GRAY)

            # save for debugging
            if False:
                cv2.imwrite("/tmp/dim/image1_gray.png", image1_gray)
                cv2.imwrite("/tmp/dim/image2_gray.png", image2_gray)
                cv2.imwrite("/tmp/dim/template_raw.png", template_raw)

            k = max_objects  # number of top results to select
            image_pd_list = []
            max_sim = np.amax(similarity)
            region_sim = []

            for i in range(k):
                ptx, pty = np.where(similarity == np.amax(similarity))
                box = [
                    int(pty[0] + 1 - (odd(template_bbox[2]) - 1) / 2),
                    int(ptx[0] + 1 - (odd(template_bbox[3]) - 1) / 2),
                    template_bbox[2],
                    template_bbox[3],
                ]
                # clip box to image size
                box[0] = max(box[0], 0)
                box[1] = max(box[1], 0)
                box[2] = min(box[2], image.shape[1])
                box[3] = min(box[3], image.shape[0])

                max_sim = 6  # By observation
                print("XXXXXX")
                print("box", box)
                sim_area = similarity[
                    box[1] : box[1] + box[3], box[0] : box[0] + box[2]
                ]

                val = np.amax(sim_area) / max_sim
                region_sim.append(val)
                print("SiM VAL ", val)
                print("-----------")

                similarity[box[1] : box[1] + box[3], box[0] : box[0] + box[2]] = 0
                image_pd_list.append(box)

            # convert to x_min, y_min, x_max, y_max
            image_pd_list = [
                [
                    max(int(round(t[0])), 0),
                    int(round(t[1])),
                    int(round(t[0] + t[2])),
                    int(round(t[1] + t[3])),
                ]
                for t in image_pd_list
            ]

            print("Predict box list:", image_pd_list)
            #  convert to x_min, y_min, w, h
            image_pd_list = [
                [
                    int(round(t[0])),
                    int(round(t[1])),
                    int(round(t[2] - t[0])),
                    int(round(t[3] - t[1])),
                ]
                for t in image_pd_list
            ]

            # image_pd_list = non_max_suppression(np.array(image_pd_list), 0.5)
            print("Predict box list:", image_pd_list)

            output = image.copy()
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template_gray = template_gray.astype(np.uint8)
            image1_gray = crop_to_content(template_gray, content_aware=True)

            predictions = []
            # find the max dimension of the two images and resize the other image to match it
            # we add 16 pixels to the max dimension to ensure that the image does not touch the border
            max_y = int(image1_gray.shape[0]) + 16
            max_x = int(image1_gray.shape[1]) + 16

            image1_gray, coord = resize_image(
                image1_gray,
                desired_size=(max_y, max_x),
                color=(255, 255, 255),
                keep_max_size=False,
            )
            blur1 = cv2.GaussianBlur(image1_gray, (7, 7), sigmaX=1.5, sigmaY=1.5)

            for idx, image_pd in enumerate(image_pd_list):
                pred_snippet = image[
                    image_pd[1] : image_pd[1] + h, image_pd[0] : image_pd[0] + w, :
                ]

                image2_gray = cv2.cvtColor(pred_snippet, cv2.COLOR_BGR2GRAY)
                image2_gray = crop_to_content(image2_gray, content_aware=True)

                if False:
                    # find the max dimension of the two images and resize the other image to match it
                    # we add 16 pixels to the max dimension to ensure that the image does not touch the border
                    max_y = int(max(image1_gray.shape[0], image2_gray.shape[0])) + 16
                    max_x = int(max(image1_gray.shape[1], image2_gray.shape[1])) + 16

                    image1_gray, coord = resize_image(
                        image1_gray,
                        desired_size=(max_y, max_x),
                        color=(255, 255, 255),
                        keep_max_size=False,
                    )

                image2_gray, coord = resize_image(
                    image2_gray,
                    desired_size=(max_y, max_x),
                    color=(255, 255, 255),
                    keep_max_size=False,
                )

                # ensure that the shapes are the same for
                if image1_gray.shape != image2_gray.shape:
                    raise ValueError(
                        f"Template and prediction snippet have different shapes: {image1_gray.shape} vs {image2_gray.shape}"
                    )

                # save for debugging
                if False:
                    stacked = np.hstack((image1_gray, image2_gray))
                    cv2.imwrite(f"/tmp/dim/image2_gray_{idx}.png", image2_gray)

                # # Calculate SSIM
                blur2 = cv2.GaussianBlur(image2_gray, (7, 7), sigmaX=1.5, sigmaY=1.5)

                # stacked = np.hstack((blur1, blur2))
                # cv2.imwrite(f"/tmp/dim/stacked_{idx}.png", stacked)

                org = blur1
                blur = blur2

                patch_size_h = image1_gray.shape[0] // 8
                patch_size_w = image1_gray.shape[1] // 8

                patch_size_h = 16
                patch_size_w = 16

                # ensure that the patch is not larger than the image
                patch_size_h = min(patch_size_h, image1_gray.shape[0])
                patch_size_w = min(patch_size_w, image1_gray.shape[1])

                patches_t = patchify(
                    blur,
                    (patch_size_h, patch_size_w),
                    step=min(patch_size_h, patch_size_w),
                )
                patches_m = patchify(
                    org,
                    (patch_size_h, patch_size_w),
                    step=min(patch_size_h, patch_size_w),
                )

                # self.viz_patches(patches_t, f"/tmp/dim/patches_t_{idx}.png")
                # self.viz_patches(patches_m, f"/tmp/dim/patches_m_{idx}.png")

                square_x = patches_t.shape[1]
                square_y = patches_t.shape[0]

                ix = 1
                s0_total = 0
                s1_total = 0

                for i in range(square_y):
                    for j in range(square_x):
                        p1 = patches_t[i, j, :, :]
                        p2 = patches_m[i, j, :, :]

                        s0 = rmse(p1, p2)  # value between 0 and 255
                        s0 = 1 - s0 / 255  # normalize rmse to 1
                        # s0 = 0

                        s1 = metrics.structural_similarity(
                            p1, p2, full=True, data_range=1
                        )[0]

                        s0_total += max(s0, 0)
                        s1_total += max(s1, 0)

                        # print(f"PP SSIM Score[{idx}]: ", round(s1, 2), s0)
                        ix += 1

                s0_total_norm = s0_total / (square_x * square_y)
                s1_total_norm = s1_total / (square_x * square_y)

                print("s0_total", s0_total)
                print("s1_total", s1_total)
                print("s0_total_norm", s0_total_norm)
                print("s1_total_norm", s1_total_norm)

                sim_val = region_sim[idx]

                score = (s0_total_norm + s1_total_norm) / 2
                score = score * sim_val

                patch_size_h = image1_gray.shape[0] // 2
                patch_size_w = image1_gray.shape[1] // 2

                patches_t = patchify(
                    blur,
                    (patch_size_h, patch_size_w),
                    step=min(patch_size_h, patch_size_w),
                )
                patches_m = patchify(
                    org,
                    (patch_size_h, patch_size_w),
                    step=min(patch_size_h, patch_size_w),
                )

                print("Patches shape: {}".format(patches_t.shape))
                print("Patches shape: {}".format(patches_m.shape))

                # self.viz_patches(patches_t, f"/tmp/dim/patches_t_{idx}.png")
                # self.viz_patches(patches_m, f"/tmp/dim/patches_m_{idx}.png")

                square_x = patches_t.shape[1]
                square_y = patches_t.shape[0]

                ix = 1
                s0_total = 0
                s1_total = 0

                for i in range(square_y):
                    for j in range(square_x):
                        p1 = patches_t[i, j, :, :]
                        p2 = patches_m[i, j, :, :]

                        s0 = rmse(p1, p2)  # value between 0 and 255
                        s0 = 1 - s0 / 255  # normalize rmse to 1
                        # s0 = 0

                        s1 = metrics.structural_similarity(
                            p1, p2, full=True, data_range=1
                        )[0]

                        s0_total += max(s0, 0)
                        s1_total += max(s1, 0)

                        # print(f"PP SSIM Score[{idx}]: ", round(s1, 2), s0)
                        ix += 1

                s0_total_norm = s0_total / (square_x * square_y)
                s1_total_norm = s1_total / (square_x * square_y)

                print("s0_total", s0_total)
                print("s1_total", s1_total)
                print("s0_total_norm", s0_total_norm)
                print("s1_total_norm", s1_total_norm)

                sim_val = region_sim[idx]

                score = (s0_total_norm + s1_total_norm) / 2
                score = score * sim_val

                print("score norm : ", score, sim_val)
                score = min(score, 1.0)

                if False:
                    # draw rectangle on image
                    cv2.rectangle(
                        output,
                        (image_pd[0], image_pd[1]),
                        (image_pd[0] + image_pd[2], image_pd[1] + image_pd[3]),
                        (0, 255, 0),
                        2,
                    )
                    # add label
                    cv2.putText(
                        output,
                        f"{round(score, 2)}",
                        (image_pd[0], image_pd[1] + image_pd[3] // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

                predictions.append(
                    {
                        "bbox": image_pd,
                        "label": template_label,
                        "score": round(score, 3),
                        "similarity": round(sim_val, 3),
                    }
                )
            # cv2.imwrite(f"/tmp/dim/output.png", output)
            return predictions


# ref https://stackoverflow.com/questions/77230340/ssim-is-large-but-the-two-images-are-not-similar-at-all
# Binary-image comparison with local-dissimilarity quantification
# https://www.sciencedirect.com/science/article/abs/pii/S0031320307003287
