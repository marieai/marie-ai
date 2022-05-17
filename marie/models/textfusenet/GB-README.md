
## Troubleshooting

Error compiling detectron2
```
  nvcc fatal   : Unsupported gpu architecture 'compute_86'
```
Follow instructions from here
https://stackoverflow.com/questions/69865825/nvcc-fatal-unsupported-gpu-architecture-compute-86


https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805

-- Could NOT find CUDNN (missing: CUDNN_LIBRARY_PATH CUDNN_INCLUDE_PATH) 
CMake Warning at cmake/public/cuda.cmake:114 (message):
  Caffe2: Cannot find cuDNN library.  Turning the option off
Call Stack (most recent call first):


## Annotation Tool
We are going to utilize total-text dataset
https://github.com/cs-chan/Total-Text-Dataset

Tool is written in matlab so we are goint to use a Octave to run it.
```
pip3 install oct2py
```

```
from oct2py import Oct2Py
oc = Oct2Py()

script = "function y = myScript(x)\n" \
         "    y = x-5" \
         "end"

with open("myScript.m","w+") as f:
    f.write(script)

oc.myScript(7)
```


# Environment setup

This version 1.10.1+cu111 causes error :
'The detected CUDA version (11.2) mismatches the version that was used to compile
    PyTorch (11.1). Please make sure to use the same CUDA versions.'
  

```
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Test 
```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

```

Works on my version 

```
1.9.0+cu111
True
```

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


=======


https://github.com/autonise/CRAFT-Remade