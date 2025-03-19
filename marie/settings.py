import os
from typing import Callable, Dict, Optional

import torch
from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    IMAGE_DPI_HIGHRES: int = 192  # Used for OCR, table rec
    ENABLE_EFFICIENT_ATTENTION: bool = (
        True  # Usually keep True, but if you get CUDA errors, setting to False can help
    )
    ENABLE_CUDNN_ATTENTION: bool = (
        False  # Causes issues on many systems when set to True, but can improve performance on certain GPUs
    )
    DISABLE_TQDM: bool = False  # Disable tqdm progress bars
    S3_BASE_URL: str = "https://models.marieai.co"
    PARALLEL_DOWNLOAD_WORKERS: int = (
        10  # Number of workers for parallel model downloads
    )

    # Paths
    DATA_DIR: str = "data"
    RESULT_DIR: str = "results"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")

    @computed_field
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        try:
            import torch_xla

            if len(torch_xla.devices()) > 0:
                return "xla"
        except:
            pass

        return "cpu"

    # Table Rec
    TABLE_REC_MODEL_CHECKPOINT: str = "s3://table_recognition/2025_02_18"
    TABLE_REC_IMAGE_SIZE: Dict = {"height": 768, "width": 768}
    TABLE_REC_MAX_BOXES: int = 150
    TABLE_REC_BATCH_SIZE: Optional[int] = None
    TABLE_REC_BENCH_DATASET_NAME: str = "datalab-to/fintabnet_bench"
    COMPILE_TABLE_REC: bool = False

    COMPILE_ALL: bool = False

    @computed_field
    def DETECTOR_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL
            or self.COMPILE_DETECTOR
            or self.TORCH_DEVICE_MODEL == "xla"
        )  # We need to static cache and pad to batch size for XLA, since it will recompile otherwise

    @computed_field
    def RECOGNITION_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL
            or self.COMPILE_RECOGNITION
            or self.TORCH_DEVICE_MODEL == "xla"
        )

    @computed_field
    def TABLE_REC_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL
            or self.COMPILE_TABLE_REC
            or self.TORCH_DEVICE_MODEL == "xla"
        )

    @computed_field
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        if self.TORCH_DEVICE_MODEL == "xla":
            return torch.bfloat16
        return torch.float16

    @computed_field
    def INFERENCE_MODE(self) -> Callable:
        if self.TORCH_DEVICE_MODEL == "xla":
            return torch.no_grad
        return torch.inference_mode

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
