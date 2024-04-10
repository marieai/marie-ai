import albumentations as aug
import torch
from torch import nn

from marie.logging.profile import TimeContext, TimeContextCuda

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
            else:
                raise ValueError("Model name must be either resnet or efficientnet")
        else:
            self.model = None

        self.transform = ToTensor(self.device)
        # self.augment = aug.Compose([aug.Normalize(p=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) ])
        self.augment = aug.Compose([aug.Normalize(p=1)])

    def get_color_features(self, image):
        # print(f"Image shape: {image.shape} GET COLOR FEATURES ")
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
        def log_cuda_time(cuda_time):
            print(f"FEATURE CUDA : {cuda_time}")
            # logger.warning(f"CUDA : {cuda_time}")
            # write to the text file
            # with open("/tmp/cuda_time_autocast_compiled.txt", "a") as f:
            #     f.write(f"{cuda_time}, {len(src_images)}, {amp_enabled}\n")

        with TimeContextCuda(
            "Feature inference", logger=None, enabled=False, callback=log_cuda_time
        ):
            with torch.no_grad():
                if self.num_features == 27:
                    image_feature = self.get_color_features(image)
                else:
                    augmented = self.augment(image=image)
                    image_norm = augmented["image"]
                    image_norm_torch = self.transform(image_norm)
                    image_feature = self.model(image_norm_torch.unsqueeze(0)).squeeze(0)

        return image_feature

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimizes the model for inference. This method is called by the __init__ method."""
        try:
            with TimeContext("Compiling model", logger=None):
                import torch._dynamo as dynamo

                torch._dynamo.config.verbose = True
                torch._dynamo.config.suppress_errors = True
                model = torch.compile(model, mode="reduce-overhead", dynamic=False)
                return model
        except Exception as err:
            raise err
