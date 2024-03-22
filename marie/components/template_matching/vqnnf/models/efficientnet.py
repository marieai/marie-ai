import os

import timm
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn

DEVICE = 'cuda'


def model_eff_b0():
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=True,
        features_only=True,
        out_indices=[0, 1, 2, 3, 4],
    )
    checkpoint = torch.load(
        os.path.expanduser(
            '~/dev/grapnel-tooling/outputs/efficientnet_b0/ef_model_pretrained_49_True.pth'
        ),
        map_location=DEVICE,
    )
    print('Loading trained model weights...')
    model_state_dict = checkpoint['model_state_dict']
    # remove the keys that are not in the model for feature extraction, we are pre-training our model on custom classes
    keys_to_remove = [
        "conv_head.weight",
        "bn2.weight",
        "bn2.bias",
        "bn2.running_mean",
        "bn2.running_var",
        "bn2.num_batches_tracked",
        "classifier.weight",
        "classifier.bias",
    ]
    for key in keys_to_remove:
        if key in model_state_dict:
            model_state_dict.pop(key)
    model.load_state_dict(model_state_dict)

    return model


def model_effv2_s():
    model = timm.create_model(
        'tf_efficientnetv2_s',
        pretrained=True,
        features_only=True,
        out_indices=[0, 1, 2, 3, 4],
    )
    checkpoint = torch.load(
        os.path.expanduser(
            '~/dev/grapnel-tooling/outputs/ef_model_pretrained_28_True.pth'
        ),
        map_location=DEVICE,
    )
    print('Loading trained model weights...')
    model_state_dict = checkpoint['model_state_dict']
    # remove the keys that are not in the model for feature extraction, we are pre-training our model on custom classes
    keys_to_remove = [
        "conv_head.weight",
        "bn2.weight",
        "bn2.bias",
        "bn2.running_mean",
        "bn2.running_var",
        "bn2.num_batches_tracked",
        "classifier.weight",
        "classifier.bias",
    ]
    for key in keys_to_remove:
        if key in model_state_dict:
            model_state_dict.pop(key)
    model.load_state_dict(model_state_dict)

    print(model)
    return model


@torch.no_grad()
def model_eff_v2_s_endpoints(model, x):
    endpoints = dict()
    output = model(x)
    for idx, tensor in enumerate(output):
        # print(f"Output {idx} shape: {tensor.shape}")
        endpoints[f"reduction_{idx + 1}"] = tensor
    return endpoints


class EfficientNetHyperColumn(nn.Module):
    def __init__(
        self, model_name, in_channels=3, num_features=256, stride=1, weights_path=None
    ):
        super().__init__()
        self.stride = stride
        self.num_features = num_features  # 40, 80, 192, 512
        self.model_type = (
            "efficientnetv2_s"  # or efficientnetv2_s efficientnet_b0_eff efficientne_b0
        )

        print(f"testing with {self.model_type}")

        if self.model_type == "efficientnet_b0_eff":
            self.model = EfficientNet.from_pretrained(
                model_name, weights_path=weights_path, advprop=False
            )
        elif self.model_type == "efficientnetv2_s":
            self.model = model_effv2_s()
        elif self.model_type == "efficientne_b0":
            self.model = model_eff_b0()

        self.model.eval()

    def extract_endpoints(self, model, x):
        if self.model_type == "efficientnet_b0":
            return model.extract_endpoints(x)
        elif (
            self.model_type == "efficientnetv2_s" or self.model_type == "efficientne_b0"
        ):
            return model_eff_v2_s_endpoints(model, x)

    def forward(self, x):
        _, _, h, w = x.shape
        out_h = h // self.stride
        out_w = w // self.stride

        out = []
        endpoints = self.extract_endpoints(self.model, x)
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
