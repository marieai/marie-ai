from typing import Mapping, Any

import torch
import torchvision.models as models
import onnxruntime

from timeit import timeit
import numpy as np
import os
from PIL import Image

# Optimizations :
#  export LD_PRELOAD=/usr/local/lib/libjemalloc.so:$LD_PRELOAD &&  python ./check_onnx_runtime.py
# https://github.com/regisss/transformers/blob/main/src/transformers/models/layoutlmv3/configuration_layoutlmv3.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3TokenizerFast,
)
from transformers.onnx import OnnxConfig
from transformers.onnx.utils import compute_effective_axis_dimension


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


def _generate_dummy_images(
    batch_size: int = 2,
    num_channels: int = 3,
    image_height: int = 40,
    image_width: int = 40,
):
    images = []
    for _ in range(batch_size):
        data = np.random.rand(image_height, image_width, num_channels) * 255
        images.append(Image.fromarray(data.astype("uint8")).convert("RGB"))
    return images


def generate_dummy_inputs(
    processor: LayoutLMv3Processor,
    batch_size: int = -1,
    seq_length: int = -1,
    is_pair: bool = False,
    num_channels: int = 3,
    image_width: int = 40,
    image_height: int = 40,
) -> Mapping[str, Any]:
    """
    Generate inputs to provide to the ONNX exporter for the specific framework
    Args:
        processor ([`ProcessorMixin`]):
            The processor associated with this model configuration.
        batch_size (`int`, *optional*, defaults to -1):
            The batch size to export the model for (-1 means dynamic axis).
        seq_length (`int`, *optional*, defaults to -1):
            The sequence length to export the model for (-1 means dynamic axis).
        is_pair (`bool`, *optional*, defaults to `False`):
            Indicate if the input is a pair (sentence 1, sentence 2).
        framework (`TensorType`, *optional*, defaults to `None`):
            The framework (PyTorch or TensorFlow) that the processor will generate tensors for.
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels of the generated images.
        image_width (`int`, *optional*, defaults to 40):
            The width of the generated images.
        image_height (`int`, *optional*, defaults to 40):
            The height of the generated images.
    Returns:
        Mapping[str, Any]: holding the kwargs to provide to the model's forward function
    """

    # A dummy image is used so OCR should not be applied
    setattr(processor.feature_extractor, "apply_ocr", False)

    # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
    batch_size = compute_effective_axis_dimension(
        batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
    )
    # If dynamic axis (-1) we forward with a fixed dimension of 8 tokens to avoid optimizations made by ONNX
    token_to_add = processor.tokenizer.num_special_tokens_to_add(is_pair)
    seq_length = compute_effective_axis_dimension(
        seq_length,
        fixed_dimension=OnnxConfig.default_fixed_sequence,
        num_token_to_add=token_to_add,
    )
    # Generate dummy inputs according to compute batch and sequence
    dummy_text = [[" ".join([processor.tokenizer.unk_token]) * seq_length]] * batch_size

    # Generate dummy bounding boxes
    dummy_bboxes = [[[48, 84, 73, 128]]] * batch_size

    # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
    batch_size = compute_effective_axis_dimension(
        batch_size, fixed_dimension=OnnxConfig.default_fixed_batch
    )
    dummy_image = _generate_dummy_images(
        batch_size, num_channels, image_height, image_width
    )

    inputs = dict(
        processor(
            dummy_image,
            text=dummy_text,
            boxes=dummy_bboxes,
            return_tensors="pt",
        )
    )

    return inputs


def export():
    print("Exporting model")

    # load model and tokenizer
    model_id = "/home/gbugaj/models/layoutlmv3-base-finetuned/checkpoint-50000"
    print(f"TokenClassification dir : {model_id}")

    # Max model size is 512, so we will need to handle any documents larger thjan ath
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    processor = LayoutLMv3Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_id)

    input_dict = generate_dummy_inputs(processor)
    print(input_dict.values())

    # https://github.com/huggingface/transformers/pull/17953
    # export
    torch.onnx.export(
        model,
        tuple(input_dict.values()),
        f="/tmp/torch-model.onnx",
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'},
        },
        do_constant_folding=True,
        opset_version=13,
    )


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = str(16)
    export()

# 222  200x200 = 40000
# 83   80x100 > 8000
