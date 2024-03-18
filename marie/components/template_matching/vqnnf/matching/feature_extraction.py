import albumentations as aug
import torch

from ..models.efficientnet import EfficientNetHyperColumn
from ..models.resnet import ResNetHyperColumn


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device):
        self.device = device

    def __call__(self, image):
        image_copy = image.transpose((2, 0, 1))
        image_torch = torch.from_numpy(image_copy).float().to(self.device)
        return image_torch


class PixelFeatureExtractor:
    def __init__(self, model_name, num_features, device="0", weights_path=None):
        self.device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )

        self.num_features = num_features
        if self.num_features != 27:
            if "resnet" in model_name:
                self.model = ResNetHyperColumn(model_name, 3, num_features).to(
                    self.device
                )
            elif "efficientnet" in model_name:
                self.model = EfficientNetHyperColumn(
                    model_name, 3, num_features, weights_path=weights_path
                ).to(self.device)
                # self.model.eval()
                # self.model.to("cpu")
                # self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
                # torch.quantization.prepare(self.model, inplace=True)
                # torch.quantization.convert(self.model, inplace=True)
                if False:
                    try:
                        import torch._dynamo as dynamo
                        import torchvision.models as models

                        self.model = torch.compile(self.model, mode="default")
                    except Exception as err:
                        print(f"Model compile not supported: {err}")
            else:
                raise ValueError("Model name must be either resnet or efficientnet")
        else:
            self.model = None

        self.transform = ToTensor(self.device)
        self.augment = aug.Compose([aug.Normalize(p=1)])

    def get_color_features(self, image):
        # get color features in 3x3 neighborhood as vector of each pixel in image
        with torch.no_grad():
            image_torch = self.transform(image) / 255
            image_feature = torch.cat(
                [
                    image_torch,
                    torch.roll(image_torch, shifts=[0, 1], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[0, -1], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[1, 0], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[-1, 0], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[1, 1], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[-1, -1], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[1, -1], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[-1, 1], dims=[1, 2]),
                ],
                dim=0,
            )
        return image_feature

    def get_features(self, image):
        with torch.no_grad():
            if self.num_features == 27:
                image_feature = self.get_color_features(image)
            else:
                self.model.eval()
                augmented = self.augment(image=image)
                image_norm = augmented["image"]
                image_norm_torch = self.transform(image_norm)
                image_feature = self.model(image_norm_torch.unsqueeze(0)).squeeze(0)

        return image_feature
