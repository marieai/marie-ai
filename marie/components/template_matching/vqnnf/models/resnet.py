from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models


def get_kwargs(model_name):
    if model_name == "resnet18":
        return {
            "layers": [2, 2, 2, 2],
            "block": BasicBlock,
            "weights": models.ResNet18_Weights.IMAGENET1K_V1,
        }
    elif model_name == "resnet34":
        return {
            "layers": [3, 4, 6, 3],
            "block": BasicBlock,
            "weights": models.ResNet34_Weights.IMAGENET1K_V1,
        }
    elif model_name == "resnet50":
        return {
            "layers": [3, 4, 6, 3],
            "block": Bottleneck,
            "weights": models.ResNet50_Weights.IMAGENET1K_V1,
        }
    elif model_name == "resnet101":
        return {
            "layers": [3, 4, 23, 3],
            "block": Bottleneck,
            "weights": models.ResNet101_Weights.IMAGENET1K_V1,
        }
    elif model_name == "resnext50_32x4d":
        return {
            "layers": [3, 4, 6, 3],
            "block": Bottleneck,
            "groups": 32,
            "width_per_group": 4,
            "weights": models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
        }
    elif model_name == "wide_resnet50_2":
        return {
            "layers": [3, 4, 6, 3],
            "block": Bottleneck,
            "groups": 32,
            "width_per_group": 64 * 2,
            "weights": models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
        }


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, model_name, in_channels=3, num_features=256, multi_scale=False):
        super(ResNet, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        kwargs = get_kwargs(model_name)
        block = kwargs["block"]
        layers = kwargs["layers"]
        self.groups = kwargs["groups"] if "groups" in kwargs else 1
        self.base_width = (
            kwargs["width_per_group"] if "width_per_group" in kwargs else 64
        )
        self.num_features = num_features
        self.multi_scale = multi_scale

        state_dict = getattr(models, model_name)(
            weights=kwargs["weights"]  # models.ResNet18_Weights.IMAGENET1K_V1
        ).state_dict()

        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)  # inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.strides = [4]
        self.inter_num_features = [64]

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

        current_num_features = 64

        if self.num_features != current_num_features:
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.inter_num_features.append(self.inplanes)
            current_num_features = (
                sum(self.inter_num_features)
                if self.multi_scale
                else self.inter_num_features[-1]
            )
            self.strides.append(4)

        if self.num_features != current_num_features:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.inter_num_features.append(self.inplanes)
            self.strides.append(8)
            current_num_features = (
                sum(self.inter_num_features)
                if self.multi_scale
                else self.inter_num_features[-1]
            )

        if self.num_features != current_num_features:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.inter_num_features.append(self.inplanes)
            self.strides.append(16)
            current_num_features = (
                sum(self.inter_num_features)
                if self.multi_scale
                else self.inter_num_features[-1]
            )
        if self.num_features != current_num_features:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.inter_num_features.append(self.inplanes)
            self.strides.append(32)
            current_num_features = (
                sum(self.inter_num_features)
                if self.multi_scale
                else self.inter_num_features[-1]
            )

        self.load_state_dict(state_dict, strict=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )

        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        return x

    def forward_ms_features(self, x):
        feat = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat.append(x)
        x = self.layer1(x)
        feat.append(x)
        if self.layer2 is not None:
            x = self.layer2(x)
            feat.append(x)
        if self.layer3 is not None:
            x = self.layer3(x)
            feat.append(x)
        if self.layer4 is not None:
            x = self.layer4(x)
            feat.append(x)
        # feat = torch.cat(feat, dim=1)
        return feat

    def forward(self, x):
        if self.multi_scale:
            return self.forward_ms_features(x)
        else:
            return self.forward_features(x)


class ResNetHyperColumn(ResNet):
    def __init__(self, model_name, in_channels=3, num_features=256, stride=1):
        super().__init__(model_name, in_channels, num_features, multi_scale=True)
        self.stride = stride

    def forward(self, x, memory_efficient=True):
        _, _, h, w = x.shape
        out_h = h // self.stride
        out_w = w // self.stride

        # out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=True)

        if not memory_efficient:
            if self.layer1 is not None:
                x = self.layer1(x)
                out.append(
                    F.interpolate(
                        x, size=(out_h, out_w), mode="bilinear", align_corners=True
                    )
                )
            if self.layer2 is not None:
                x = self.layer2(x)
                out.append(
                    F.interpolate(
                        x, size=(out_h, out_w), mode="bilinear", align_corners=True
                    )
                )
            if self.layer3 is not None:
                x = self.layer3(x)
                out.append(
                    F.interpolate(
                        x, size=(out_h, out_w), mode="bilinear", align_corners=True
                    )
                )
            if self.layer4 is not None:
                x = self.layer4(x)
                out.append(
                    F.interpolate(
                        x, size=(out_h, out_w), mode="bilinear", align_corners=True
                    )
                )
            out = torch.cat(out, dim=1)
        else:
            if self.layer1 is not None:
                x = self.layer1(x)
                out = torch.cat(
                    [
                        out,
                        F.interpolate(
                            x, size=(out_h, out_w), mode="bilinear", align_corners=True
                        ),
                    ],
                    dim=1,
                )
            if self.layer2 is not None:
                x = self.layer2(x)
                out = torch.cat(
                    [
                        out,
                        F.interpolate(
                            x, size=(out_h, out_w), mode="bilinear", align_corners=True
                        ),
                    ],
                    dim=1,
                )
            if self.layer3 is not None:
                x = self.layer3(x)
                out = torch.cat(
                    [
                        out,
                        F.interpolate(
                            x, size=(out_h, out_w), mode="bilinear", align_corners=True
                        ),
                    ],
                    dim=1,
                )
            if self.layer4 is not None:
                x = self.layer4(x)
                out = torch.cat(
                    [
                        out,
                        F.interpolate(
                            x, size=(out_h, out_w), mode="bilinear", align_corners=True
                        ),
                    ],
                    dim=1,
                )
        return out
