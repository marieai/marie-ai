import decimal
import time
from typing import List, Optional, Tuple, Union

import colorcet as cc
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.color import label2rgb
from torch import nn

# from fast_pytorch_kmeans import KMeans
from .kmeans import KMeans
from .utils import convert_box_to_integral, get_center_crop_coords

HAAR_CONF_DICT = {
    "haar_1x": {
        "weights": torch.tensor([[1]]).float(),
        "dilation_x": 1,
        "dilation_y": 1,
        "filter_weight": 1,
    },
    "haar_2x": {
        "weights": torch.tensor([[1, -1]]).float(),
        "dilation_x": 1,
        "dilation_y": 2,
        "filter_weight": 0.25,
    },
    "haar_2y": {
        "weights": torch.tensor([[1], [-1]]).float(),
        "dilation_x": 2,
        "dilation_y": 1,
        "filter_weight": 0.25,
    },
    "haar_3x": {
        "weights": torch.tensor([[-1, 2, -1]]).float(),
        "dilation_x": 1,
        "dilation_y": 3,
        "filter_weight": 0.25,
    },
    "haar_3y": {
        "weights": torch.tensor([[-1], [2], [-1]]).float(),
        "dilation_x": 3,
        "dilation_y": 1,
        "filter_weight": 0.25,
    },
    "haar_4xy": {
        "weights": torch.tensor([[1, -1], [-1, 1]]).float(),
        "dilation_x": 2,
        "dilation_y": 2,
        "filter_weight": 1.00,
    },
}


def get_gaussian_box_filter(kernel_size, sigma, order=(0, 0)):
    kernel = np.zeros(kernel_size)
    center_x = kernel_size[0] // 2
    center_y = kernel_size[1] // 2
    if kernel_size[0] % 2 == 0:
        x_ones = [center_x - 1, center_x + 1]
        x_mult = 0.5
    else:
        x_ones = [center_x, center_x + 1]
        x_mult = 1
    if kernel_size[1] % 2 == 0:
        y_ones = [center_y - 1, center_y + 1]
        y_mult = 0.5
    else:
        y_ones = [center_y, center_y + 1]
        y_mult = 1
    kernel[x_ones[0] : x_ones[1], y_ones[0] : y_ones[1]] = x_mult * y_mult
    kernel = ndimage.gaussian_filter(kernel, sigma=sigma, order=order)
    return torch.tensor(kernel).float()


def get_gauss_haar_integral(kernel_size, sigma, haar_filter_dict):
    haar_weight = haar_filter_dict["weights"]
    haar_weight_shape = haar_weight.shape
    new_kernel_shape_x = (
        kernel_size[0]
        if kernel_size[0] % haar_weight_shape[0] == 0
        else (kernel_size[0] // haar_weight_shape[0] + 1) * haar_weight_shape[0]
    )
    new_kernel_shape_y = (
        kernel_size[1]
        if kernel_size[1] % haar_weight_shape[1] == 0
        else (kernel_size[1] // haar_weight_shape[1] + 1) * haar_weight_shape[1]
    )

    gaussian_filter = get_gaussian_box_filter(
        (new_kernel_shape_x, new_kernel_shape_y), sigma
    )
    repeat_x = new_kernel_shape_x // haar_weight_shape[0]
    repeat_y = new_kernel_shape_y // haar_weight_shape[1]
    haar_filter = haar_weight.repeat_interleave(repeat_x, axis=0).repeat_interleave(
        repeat_y, axis=1
    )

    haar_filter = haar_filter * gaussian_filter
    # print(haar_filter)
    haar_integral_filter = convert_box_to_integral(haar_filter)
    return haar_integral_filter


class GaussHaarFilters(nn.Module):
    def __init__(
        self,
        n_channels: int,
        template_shape: Tuple[int, int],
        kernel_size: Union[Tuple[int, int], int] = (3, 3),
        sigma: Union[Tuple[int, int], int] = (2, 2),
        filters: int = 1,
        n_scales: int = 1,
        scale_weights: Optional[List[float]] = None,
        channel_weights: Optional[List[float]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.n_channels = n_channels
        self.channel_weights = (
            torch.from_numpy(np.array(channel_weights)).float().to(self.device)
        )
        self.template_shape = template_shape

        rects_to_consider = decimal.Decimal(filters).as_tuple().digits
        filters_to_consider = ["haar_1x"]
        filters_to_consider.extend(
            [
                k
                for k in HAAR_CONF_DICT.keys()
                if int(list(filter(str.isdigit, k))[0]) in rects_to_consider
            ]
        )

        self.n_filters = len(filters_to_consider) * n_scales

        self.filter_kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.sigmas = (sigma, sigma) if isinstance(sigma, int) else sigma

        scales_to_consider = np.linspace(1, 1 / n_scales, n_scales)
        if scale_weights is None:
            scale_weights = scales_to_consider
        assert n_scales == len(
            scale_weights
        ), "Number of scales and scale weights should be the same"

        self.filter_names = []
        self.filters = []
        self.filter_weights = []
        self.scale_kernel_sizes = []
        for scale_to_consider, scale_weight in zip(scales_to_consider, scale_weights):
            self.get_filters(
                n_channels,
                template_shape,
                filters_to_consider,
                scale_to_consider,
                scale_weight,
            )

        self.template_features = []

    def get_filters(
        self,
        n_channels: int,
        kernel_size: Tuple[int, int],
        filters_to_consider: List[str],
        scale: float,
        scale_weight: float,
    ):
        w = int(kernel_size[0] * scale)
        h = int(kernel_size[1] * scale)

        for filter_name in filters_to_consider:
            self.filter_names.append(f"{scale}_{filter_name}")

            filter_config = HAAR_CONF_DICT[filter_name]
            self.filter_weights.append(scale_weight * filter_config["filter_weight"])

            integral_filter_kernel = get_gauss_haar_integral(
                self.filter_kernel_size, self.sigmas, filter_config
            )
            filter_kernel_size = integral_filter_kernel.shape
            dilation = (
                (w) // (filter_kernel_size[0] - 1),
                (h) // (filter_kernel_size[1] - 1),
            )
            w_kernel = (filter_kernel_size[0] - 1) * dilation[0] + 1
            h_kernel = (filter_kernel_size[1] - 1) * dilation[1] + 1
            self.scale_kernel_sizes.append((w_kernel, h_kernel))

            filter = nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=filter_kernel_size,
                dilation=dilation,
                bias=False,
                groups=n_channels,
            )
            filter.weight.data = integral_filter_kernel[None, None].repeat(
                n_channels, 1, 1, 1
            ) / (filter_kernel_size[0] * filter_kernel_size[1])
            filter.weight.requires_grad = False
            self.filters.append(filter.to(self.device))

    def forward_filter(
        self,
        x: torch.Tensor,
        filter: torch.nn.Conv2d,
        kernel_size: Tuple[int, int],
        out_shape: Optional[Tuple[int, int]] = None,
    ):
        with torch.no_grad():
            pad_x = max(0, np.ceil((kernel_size[0] - x.shape[2]) / 2).astype(int))
            pad_y = max(0, np.ceil((kernel_size[1] - x.shape[3]) / 2).astype(int))
            x_pad = F.pad(x, (pad_y, pad_y, pad_x, pad_x), mode="reflect")
            y_pad = filter(x_pad)
            if out_shape is not None:
                x1, y1, x2, y2 = get_center_crop_coords(
                    y_pad.shape[2], y_pad.shape[3], out_shape[0], out_shape[1]
                )
                y = y_pad[:, :, x1:x2, y1:y2]
            else:
                y = y_pad
            return y

    def get_template_features(self, template: torch.Tensor):
        for filter, kernel_size in zip(self.filters, self.scale_kernel_sizes):
            y = self.forward_filter(template, filter, kernel_size, (1, 1))
            self.template_features.append(y)
            # print(y.abs().max().item(), y.abs().min().item())

    def get_query_map(
        self, x: torch.Tensor, channel_weights: Optional[np.array] = None
    ):
        channel_weights = (
            self.channel_weights if channel_weights is None else channel_weights
        )

        distrib_sim = torch.zeros(
            x.shape[2:], device=self.device
        )  # np.zeros(x.shape[2:])
        all_filt_sims = []

        for i, (filter, kernel_size) in enumerate(
            zip(self.filters, self.scale_kernel_sizes)
        ):
            y = self.forward_filter(x, filter, kernel_size, out_shape=None)

            filter_shift = (
                torch.abs(y - self.template_features[i])
                * channel_weights[None, :, None, None]
            )
            filter_sim = -filter_shift.sum(dim=1).squeeze(0) * self.filter_weights[i]

            pad_x_left = (distrib_sim.shape[0] - y.shape[2]) // 2
            pad_x_right = distrib_sim.shape[0] - y.shape[2] - pad_x_left
            pad_y_left = (distrib_sim.shape[1] - y.shape[3]) // 2
            pad_y_right = distrib_sim.shape[1] - y.shape[3] - pad_y_left

            filter_sim = F.pad(
                filter_sim,
                (pad_y_left, pad_y_right, pad_x_left, pad_x_right),
                mode="constant",
                value=filter_sim.min().item(),
            )  # .cpu().numpy()

            distrib_sim += filter_sim
            # all_filt_sims.append(filter_sim)

        return distrib_sim, all_filt_sims
