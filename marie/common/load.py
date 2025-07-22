from typing import Any, Optional

import torch

from marie.settings import settings


class ModelLoader:
    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint

    def model(
        self,
        device: torch.device | str | None = settings.TORCH_DEVICE_MODEL,
        dtype: Optional[torch.dtype | str] = settings.MODEL_DTYPE,
    ) -> Any:
        raise NotImplementedError()

    def processor(self) -> Any:
        raise NotImplementedError()
