from timeit import timeit

import numpy as np
import onnx
import onnxruntime
import psutil
import torch
import torchvision.models as models
import transformers

# Optimizations :
#  export LD_PRELOAD=/usr/local/lib/libjemalloc.so:$LD_PRELOAD &&  python ./check_onnx_runtime.py
# Docs
# https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb
# https://onnx.ai/index.html

print("pytorch:", torch.__version__)
print("onnxruntime:", onnxruntime.__version__)
print("onnx:", onnx.__version__)
print("transformers:", transformers.__version__)


def save_temp_models(device):
    model = models.resnet50(pretrained=True)
    model.eval()
    model.to(device)

    # PyTorch model
    torch.save(model, "resnet.pth")
    # random input
    data = torch.rand(1, 3, 512, 512)
    data = data.cuda()
    # ONNX needs data example
    torch.onnx.export(model, data, "resnet.onnx")


if __name__ == "__main__":
    assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
    device_name = 'cuda'

    save_temp_models(device_name)
    import os

    os.environ["OMP_NUM_THREADS"] = str(16)

    sess_options = onnxruntime.SessionOptions()
    # Please change the value according to best setting in Performance Test Tool result.
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

    # PyTorch model
    torch_model = torch.load("resnet.pth")
    torch_model.eval()

    # ONNX model
    onnx_model = onnxruntime.InferenceSession(
        "resnet.onnx",
        sess_options,
        providers=[
            # 'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            # 'CPUExecutionProvider',
        ],
    )

    data = np.random.rand(1, 3, 512, 512).astype(np.float32)

    torch_data = torch.from_numpy(data)
    torch_data = torch_data.cuda()

    def torch_inf():
        torch_output = torch_model(torch_data)

    def onnx_inf():
        ort_outputs = onnx_model.run(None, {onnx_model.get_inputs()[0].name: data})

    n = 100
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
