import os
from typing import List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

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
        model = models.vgg19(weights=models.vgg.VGG19_Weights.IMAGENET1K_V1)
        # model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
        # model = models.vgg11()

        model = model.features
        model.eval()
        model.to(self.device)

        return model

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

        featex = self.featex
        layer1, layer2, layer3 = self.layers
        image = frame
        image_plot = image.copy()

        for template_raw in templates:
            template_bbox = [
                0,
                0,
                template_raw.shape[1],
                template_raw.shape[0],
            ]  # x, y, w, h

            x, y, w, h = [int(round(t)) for t in template_bbox]
            template_plot = cv2.rectangle(
                template_raw.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            image_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
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
                I_feat = featex(T_image, 'big')
                SI_feat = featex(T_search_image, 'big')
                resize_bbox = [i / b for i in template_bbox]

            print(' ')
            print('Feature extraction done.')
            pad1 = [int(round(t)) for t in (template_bbox[3], template_bbox[2])]
            pad2 = [int(round(t)) for t in (resize_bbox[3], resize_bbox[2])]

            SI_pad = torch.from_numpy(
                np.pad(
                    SI_feat.cpu().numpy(),
                    ((0, 0), (0, 0), (pad2[0], pad2[0]), (pad2[1], pad2[1])),
                    'symmetric',
                )
            )
            similarity = apply_DIM(I_feat, SI_pad, resize_bbox, pad2, pad1, image, 10)

            ptx, pty = np.where(similarity == np.amax(similarity))
            image_pd = tuple(
                [
                    pty[0] + 1 - (odd(template_bbox[2]) - 1) / 2,
                    ptx[0] + 1 - (odd(template_bbox[3]) - 1) / 2,
                    template_bbox[2],
                    template_bbox[3],
                ]
            )
            print('Predict box:', image_pd)

            # Plotting
            x, y, w, h = [int(round(t)) for t in image_pd]
            image_plot = cv2.rectangle(
                image_plot, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2
            )

            fig, ax = plt.subplots(1, 3, figsize=(20, 5))
            plt.ion()
            ax[0].imshow(template_plot)
            ax[1].imshow(image_plot)
            ax[2].imshow(similarity, 'jet')

            plt.savefig('/tmp/dim/results.png')
            plt.pause(0.0001)
            plt.close()
            print('Done, results saved')
