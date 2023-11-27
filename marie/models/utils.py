import os
from typing import Optional, List, Union, Tuple

import logging

import psutil
import torch

from marie.utils.types import strtobool

logger = logging.getLogger(__name__)


def initialize_device_settings(
    use_cuda: Optional[bool] = None,
    local_rank: int = -1,
    multi_gpu: bool = True,
    devices: Optional[List[Union[str, torch.device]]] = None,
) -> Tuple[List[torch.device], int]:
    """
    Returns a list of available devices.

    :param use_cuda: Whether to make use of CUDA GPUs (if available).
    :param local_rank: Ordinal of device to be used. If -1 and `multi_gpu` is True, all devices will be used.
                       Unused if `devices` is set or `use_cuda` is False.
    :param multi_gpu: Whether to make use of all GPUs (if available).
                      Unused if `devices` is set or `use_cuda` is False.
    :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
    """
    if (
        use_cuda is False
    ):  # Note that it could be None, in which case we also want to just skip this step.
        devices_to_use = [torch.device("cpu")]
        n_gpu = 0
    elif devices:
        if not isinstance(devices, list):
            raise ValueError(
                f"devices must be a list, but got {devices} of type {type(devices)}"
            )
        if any(isinstance(device, str) for device in devices):
            torch_devices: List[torch.device] = [
                torch.device(device) for device in devices
            ]
            devices_to_use = torch_devices
        else:
            devices_to_use = devices  # type: ignore [assignment]
        n_gpu = sum(1 for device in devices_to_use if "cpu" not in device.type)
    elif local_rank == -1:
        if torch.cuda.is_available():
            if multi_gpu:
                devices_to_use = [
                    torch.device(device) for device in range(torch.cuda.device_count())
                ]
                n_gpu = torch.cuda.device_count()
            else:
                devices_to_use = [torch.device("cuda:0")]
                n_gpu = 1
        else:
            devices_to_use = [torch.device("cpu")]
            n_gpu = 0
    else:
        devices_to_use = [torch.device("cuda", local_rank)]
        torch.cuda.set_device(devices_to_use[0])
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    # HF transformers v4.21.2 pipeline object doesn't accept torch.device("cuda"), it has to be an indexed cuda device
    # TODO eventually remove once the limitation is fixed in HF transformers
    device_to_replace = torch.device("cuda")
    devices_to_use = [
        torch.device("cuda:0") if device == device_to_replace else device
        for device in devices_to_use
    ]

    # sometimes we have CUDA/GPU support but want to only use CPU
    if os.environ.get("MARIE_DISABLE_CUDA"):
        devices_to_use = [torch.device("cpu")]
        n_gpu = 0

    logger.info(
        "Using devices: %s - Number of GPUs: %s",
        ", ".join([str(device) for device in devices_to_use]).upper(),
        n_gpu,
    )
    return devices_to_use, n_gpu


def enable_tf32():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def torch_gc():
    """Run torch garbage collection and CUDA IPC collect."""
    if torch.cuda.is_available():
        for devid in range(torch.cuda.device_count()):
            device_id = f"cuda:{devid}"
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


################################################################
# OpenMP setup
################################################################


def openmp_setup(threads: int):
    """Set OpenMP environment variables.

    Arguments:
        threads (int): number of threads
    """
    logger.info(f"Setting OMP_NUM_THREADS to {threads}")

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_SCHEDULE"] = "STATIC"
    os.environ["OMP_PROC_BIND"] = "CLOSE"
    os.environ["OMP_PLACES"] = "CORES"


def setup_torch_optimizations(num_threads: int = -1):
    """
    Setup torch optimizations
    :return:
    """
    try:
        logger.info(f"Setting up torch optimizations")

        if strtobool(os.environ.get("MARIE_SKIP_TORCH_OPTIMIZATION", False)):
            logger.info("Skipping torch optimizations")
            return

        # Optimizations for PyTorch
        core_count = num_threads
        if num_threads == -1:
            core_count = psutil.cpu_count(logical=False)

        torch_versions = torch.__version__.split(".")
        torch_major_version = int(torch_versions[0])
        torch_minor_version = int(torch_versions[1])
        if torch_major_version > 1 or (
            torch_major_version == 1 and torch_minor_version >= 12
        ):
            # Gives a large speedup on Ampere-class GPUs
            torch.set_float32_matmul_precision("high")

        logger.info(f"Setting up TF32")
        enable_tf32()

        logger.info(f"Setting up OpenMP with {core_count} threads")
        openmp_setup(core_count)
        torch.set_num_threads(core_count)

        # Enable oneDNN Graph, which is a graph optimization pass that fuses multiple operators into a single kernel
        torch.jit.enable_onednn_fusion(True)
    except Exception as e:
        raise e
