from timeit import timeit

import numpy as np
import onnx
import onnxruntime
import psutil
import torch
import torchvision.models as models
import transformers


import os
import torch
import onnx
import torchvision.models as models
import onnxruntime
import time


# Optimizations :
#  export LD_PRELOAD=/usr/local/lib/libjemalloc.so:$LD_PRELOAD &&  python ./check_onnx_runtime.py
# Docs
# https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb
# https://onnx.ai/index.html

print("pytorch:", torch.__version__)
print("onnxruntime:", onnxruntime.__version__)
print("onnx:", onnx.__version__)
print("transformers:", transformers.__version__)


batch_size = 1
total_samples = 1000
device = torch.device('cuda:0')

def convert_to_onnx(resnet):
   resnet.eval()
   dummy_input = (torch.randn(batch_size, 3, 224, 224, device=device)).to(device=device)
   input_names = [ 'input' ]
   output_names = [ 'output' ]
   torch.onnx.export(resnet, 
               dummy_input,
               "resnet18.onnx",
               verbose=True,
               opset_version=13,
               input_names=input_names,
               output_names=output_names,
               export_params=True,
               do_constant_folding=True,
               dynamic_axes={
                  'input': {0: 'batch_size'},  # variable length axes
                  'output': {0: 'batch_size'}}
               )

def infer_pytorch(resnet):
   print('Pytorch Inference')
   print('==========================')
   print()

   x = torch.randn((batch_size, 3, 224, 224))
   x = x.to(device=device)

   latency = []
   for i in range(total_samples):
      t0 = time.time()
      resnet.eval()
      with torch.no_grad():
         out = resnet(x)
      latency.append(time.time() - t0)

   print('Number of runs:', len(latency))
   print("Average PyTorch {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f')))  

def to_numpy(tensor):
   return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def infer_onnxruntime():
   print('Onnxruntime Inference')
   print('==========================')
   print()

   onnx_model = onnx.load("resnet18.onnx")
   onnx.checker.check_model(onnx_model)

   # Input
   x = torch.randn((batch_size, 3, 224, 224))
   x = x.to(device=device)
   x = to_numpy(x)

   torch.backends.cudnn.benchmark = False

   so = onnxruntime.SessionOptions()
   exproviders = ["CUDAExecutionProvider", "CPUExecutionProvider"]
   # HEURISTIC  DEFAULT  EXHAUSTIVE 
#    exproviders=[ ( "CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": "1", "cudnn_conv_algo_search": "DEFAULT"}),        "CPUExecutionProvider"]

   exproviders =[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"}), "CPUExecutionProvider"]

   model_onnx_path = os.path.join(".", "resnet18.onnx")
   ort_session = onnxruntime.InferenceSession(model_onnx_path, so, providers=exproviders)

   #IOBinding
   input_names = ort_session.get_inputs()[0].name
   output_names = ort_session.get_outputs()[0].name
   io_binding = ort_session.io_binding()

   io_binding.bind_cpu_input(input_names, x)
   io_binding.bind_output(output_names, 'cuda')

   #warm up run
   ort_session.run_with_iobinding(io_binding)
   ort_outs = io_binding.copy_outputs_to_cpu()

   latency = []

   for i in range(total_samples):
      t0 = time.time()
      ort_session.run_with_iobinding(io_binding)
      latency.append(time.time() - t0)
      ort_outs = io_binding.copy_outputs_to_cpu()
   print('Number of runs:', len(latency))
   print("Average onnxruntime {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f')))   

if __name__ == '__main__':
   torch.cuda.empty_cache()
   resnet = (models.resnet18(pretrained=True)).to(device=device)
   convert_to_onnx(resnet)
   infer_onnxruntime()
   infer_pytorch(resnet)