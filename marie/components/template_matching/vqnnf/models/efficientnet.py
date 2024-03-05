import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn


class EfficientNetHyperColumn(nn.Module):
    def __init__(
        self, model_name, in_channels=3, num_features=256, stride=1, weights_path=None
    ):
        super().__init__()
        self.stride = stride
        self.num_features = num_features  # 40, 80, 192, 512
        self.model = EfficientNet.from_pretrained(
            model_name, weights_path=weights_path, advprop=False
        )

    def forward(self, x):
        _, _, h, w = x.shape
        out_h = h // self.stride
        out_w = w // self.stride

        out = []
        endpoints = self.model.extract_endpoints(x)
        if self.num_features == 40:
            out.append(
                F.interpolate(
                    endpoints["reduction_1"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            out.append(
                F.interpolate(
                    endpoints["reduction_2"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
        elif self.num_features == 80:
            out.append(
                F.interpolate(
                    endpoints["reduction_1"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            out.append(
                F.interpolate(
                    endpoints["reduction_2"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            out.append(
                F.interpolate(
                    endpoints["reduction_3"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
        elif self.num_features == 192:
            out.append(
                F.interpolate(
                    endpoints["reduction_1"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            out.append(
                F.interpolate(
                    endpoints["reduction_2"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            out.append(
                F.interpolate(
                    endpoints["reduction_3"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            out.append(
                F.interpolate(
                    endpoints["reduction_4"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
        elif self.num_features == 512:
            out.append(
                F.interpolate(
                    endpoints["reduction_1"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            out.append(
                F.interpolate(
                    endpoints["reduction_2"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            out.append(
                F.interpolate(
                    endpoints["reduction_3"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            out.append(
                F.interpolate(
                    endpoints["reduction_4"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            out.append(
                F.interpolate(
                    endpoints["reduction_5"],
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        out = torch.cat(out, dim=1)
        # out = out / torch.norm(out, dim=1, keepdim=True)
        return out
