---
sidebar_position: 3
---

# Optimizing models for inference with ONNX

Marie-AI supports the [ONNX](https://onnx.ai/) format for inference. ONNX is an open format for machine learning models that allows you to easily move models between different frameworks. 

# Exporting models to ONNX
There are many ways to export models to ONNX. Here we will show you how to export our PyTorch model to ONNX.

## TensorRT support
TensorRT is a high-performance deep learning inference platform that delivers low latency and high-throughput for deep learning inference applications.


### Setup
We need to setup the environment to support TensorRT. 



## Optimize Pix2Pix models export for document overlay 
In the tools directory we have a script that will export the Pix2Pix model to ONNX. 

```sh
python3 tools/convert_onnx_overlay.py \
    --model_path /mnt/data/marie-ai/model_zoo/pix2pix/pix2pix.pth \
    --output_path /mnt/data/marie-ai/model_zoo/pix2pix/pix2pix.onnx
```


# Optimize Fairseq models for OCR
Fairseq is used by Marie-AI for OCR as it is based on the TrOCR model.

We have a custom build of Fairseq that supports ONNX export and inference. 
You can find the source code [here](https://github.com/marieai/fairseq)

To optimize Fairseq workflow we have to do the following:
-     Divide model into Encoder and Decoder two parts, and separately export to onnx model.
-     Because of the model structure define by input seq_len, should export dynamic shape onnx model.
-     Replace the Fairseq TextRecognitionGenerator task pipeline Encoder and Decoder into ONNX inference model.

**Encoder and Decoder:**
* Encoder is for extracting feature information from image.
* Decoder is for decoding the feature information to generate text information.

## Requirement
Checkout the source code and install the package in editable mode.
Reference: [GitHub: Fairseq-MarieAI](https://github.com/marieai/fairseq.git)

### Install fairseq and requirement
```sh
git clone https://github.com/marieai/fairseq.git
cd fairseq
pip install --editable .
```

Edit `sequence_generator.py` to apply changes to the source code.


# References
- [ONNX](https://onnx.ai/)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [openvino](https://blog.openvino.ai/blog-posts/openvino-tm-optimizer-fairseq-s2t-model)


https://github.com/facebookresearch/fairseq/issues/1669
https://github.com/18582088138/fairseq-openvino/blob/bc61ffe59ae79870815d34d2664a9fffe6d9c694/fairseq/sequence_generator.py