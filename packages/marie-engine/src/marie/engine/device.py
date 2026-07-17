"""Device selection shared by local Marie engine providers."""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def initialize_device_settings(
    use_cuda: Optional[bool] = None,
    local_rank: int = -1,
    multi_gpu: bool = True,
    devices: Optional[list[str | torch.device]] = None,
) -> tuple[list[torch.device], int]:
    """Resolve the devices available to a local engine provider."""
    if use_cuda is False:
        devices_to_use = [torch.device("cpu")]
        gpu_count = 0
    elif devices:
        devices_to_use = [
            torch.device(device) if isinstance(device, str) else device
            for device in devices
        ]
        gpu_count = sum(device.type != "cpu" for device in devices_to_use)
    elif local_rank == -1:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count() if multi_gpu else 1
            devices_to_use = [
                torch.device("cuda", index) for index in range(device_count)
            ]
            gpu_count = device_count
        else:
            devices_to_use = [torch.device("cpu")]
            gpu_count = 0
    else:
        devices_to_use = [torch.device("cuda", local_rank)]
        torch.cuda.set_device(devices_to_use[0])
        torch.distributed.init_process_group(backend="nccl")
        gpu_count = 1

    devices_to_use = [
        torch.device("cuda:0") if device == torch.device("cuda") else device
        for device in devices_to_use
    ]
    if os.environ.get("MARIE_DISABLE_CUDA"):
        devices_to_use = [torch.device("cpu")]
        gpu_count = 0

    logger.info("Using devices: %s - Number of GPUs: %s", devices_to_use, gpu_count)
    return devices_to_use, gpu_count
