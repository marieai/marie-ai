
# Reinstal 2.5.1
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip3 install xformers==0.0.29
```


# pip install --upgrade opencv-python
# pip install  --upgrade torchvision
# pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124

# ISSUES TO RESOLVE BEFORE UPGRADING TO TORCH 2.6

2025-03-25 15:59:29,310 - xformers - WARNING - WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
    PyTorch 2.5.1+cu121 with CUDA 1201 (you have 2.6.0+cu124)
    Python  3.12.7 (you have 3.12.3)
  Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
  Memory-efficient attention, SwiGLU, sparse and more won't be available.
  Set XFORMERS_MORE_DETAILS=1 for more details


ERROR  2025-03-25 16:05:34,954:main        : MARIE@87631 Error setting up cache : Weights only load                     
       failed. This file can still be loaded, to do so you have two options, do those steps only                     
       if you trust the source of the checkpoint.                                                                    
               (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in                       
       `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to                          
       `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you                    
       got the file from a trusted source.                                                                              
               (2) Alternatively, to load with `weights_only=True` please check the recommended                         
       steps in the following error message.                                                                            
               WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace was not an                         
       allowed global by default. Please use `torch.serialization.add_safe_globals([Namespace])` or                     
       the `torch.serialization.safe_globals([Namespace])` context manager to allowlist this global                     
       if you trust this class/function.                                                                                
                                       