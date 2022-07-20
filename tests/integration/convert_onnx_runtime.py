import torch
import torchvision.models as models
import onnxruntime

from timeit import timeit
import numpy as np
import os

# Optimizations :
#  export LD_PRELOAD=/usr/local/lib/libjemalloc.so:$LD_PRELOAD &&  python ./check_onnx_runtime.py


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def profile():
    # PyTorch model
    torch_model = torch.load("resnet.pth")
    # ONNX model
    onnx_model = onnxruntime.InferenceSession("resnet.onnx")

    data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    torch_data = torch.from_numpy(data)

    def torch_inf():
        torch_model(torch_data)

    def onnx_inf():
        onnx_model.run(None, {onnx_model.get_inputs()[0].name: data})

    n = 20
    torch_t = timeit(lambda: torch_inf(), number=n) / 100
    onnx_t = timeit(lambda: onnx_inf(), number=n) / 100
    rat = 1 - (onnx_t / torch_t)
    print(f"PyTorch {torch_t} VS ONNX {onnx_t}")
    print(f"Improvement {rat} ")

    import matplotlib.pyplot as plt

    plt.figure()

    frameworks = ["PyTorch", "ONNX"]
    times = [torch_t, onnx_t]

    plt.bar(frameworks[0], times[0])
    plt.bar(frameworks[1], times[1])
    plt.show()

def export():
    print("Exporting model")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = str(16)
    export()