import math
import os
import time
from enum import Enum
from pathlib import Path
from typing import List, Union

import torch
import torch.fx as fx
import torch.nn.utils.prune
from tqdm import tqdm


class Precision(Enum):
    int8 = "int8"
    float16 = "float16"


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


class TensorType(str, Enum):
    FLOAT = "float"
    INT = "int"
    LONG = "long"


class Architecture(str, Enum):
    ipex = "ipex"
    tensorrt = "tensorrt"
    fastertransformer = "fastertransformer"


def load_model(model_path: str, device="cpu") -> torch.nn.Module:
    map_location = torch.device(device)
    model = torch.load(model_path, map_location=map_location)
    return model


def print_size_of_model(model: torch.nn.Module, label: str = ""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, ':', 'Size (MB):', size / 1e6)
    os.remove('temp.p')
    return size


if __name__ == "__main__":
    if not os.path.exists("models_optimized/"):
        os.makedirs("models_optimized")
    model = load_model("../model_zoo/textfusenet/model_0153599.pth")
    print(model)
    print_size_of_model(model)
