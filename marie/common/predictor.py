from typing import Optional

import torch
import torch.nn.functional as F

from marie.common.load import ModelLoader
from marie.settings import settings


# TODO : migrate to  marie.base_handler import BaseHandler
class BasePredictor:
    model_loader_cls = ModelLoader
    batch_size: Optional[int] = None
    default_batch_sizes = {"cpu": 1, "mps": 1, "cuda": 1}
    disable_tqdm: bool = settings.DISABLE_TQDM

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: torch.device | str | None = settings.TORCH_DEVICE_MODEL,
        dtype: Optional[torch.dtype | str] = settings.MODEL_DTYPE,
    ):
        self.model = None
        self.processor = None
        loader = self.model_loader_cls(checkpoint)

        self.model = loader.model(device, dtype)
        self.processor = loader.processor()

    def to(self, device_dtype: torch.device | str | None = None):
        if self.model:
            self.model.to(device_dtype)
        else:
            raise ValueError("Model not loaded")

    def get_batch_size(self):
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = self.default_batch_sizes["cpu"]
            if settings.TORCH_DEVICE_MODEL in self.default_batch_sizes:
                batch_size = self.default_batch_sizes[settings.TORCH_DEVICE_MODEL]
        return batch_size

    @staticmethod
    def pad_to_batch_size(tensor: torch.Tensor, batch_size: int):
        current_batch_size = tensor.shape[0]
        if current_batch_size >= batch_size:
            return tensor

        pad_size = batch_size - current_batch_size
        padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)

        return F.pad(tensor, padding, mode='constant', value=0)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
