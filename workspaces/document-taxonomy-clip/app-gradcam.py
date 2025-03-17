import os
from time import time
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# TODO: Move this to document_taxonomy package
use_cuda = torch.cuda.is_available()

# Load two different CLIP models
model1_name = "openai/clip-vit-base-patch16"
model2_name = "openai/clip-vit-large-patch14"
model2_name = "openai/clip-vit-large-patch14-336"
model2_name = (
    "/home/gbugaj/dev/grapnel-tooling/train-clip/oputput-taxonomy/checkpoint-5000"
)

# Load CLIP models and their respective processors
model_1 = CLIPModel.from_pretrained(model1_name)
processor_1 = CLIPProcessor.from_pretrained(model1_name)

model_2 = CLIPModel.from_pretrained(model2_name)
processor_2 = CLIPProcessor.from_pretrained(model2_name)


def get_gradcam_activations(model, image_tensor, text_tensor):
    """
    Extract Grad-CAM activations for the vision backbone in the CLIP model.

    Args:
        model: Pre-trained CLIP model.
        image_tensor: Input tensor for the image.
        text_tensor: Input tensor for the text.

    Returns:
        activations: The captured feature map activations.
        gradients: The gradients with respect to the activations.
    """
    activations = None
    gradients = None

    # Define hooks to capture gradients and activations
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # Access the final encoder layer of the vision model
    target_layer = model.vision_model.encoder.layers[
        -1
    ]  # Last encoder layer in the vision transformer

    # Register hooks for forward and backward passes
    forward_hook_handle = target_layer.self_attn.register_forward_hook(forward_hook)
    backward_hook_handle = target_layer.self_attn.register_backward_hook(backward_hook)

    # Forward pass to compute outputs
    inputs = {'pixel_values': image_tensor, 'input_ids': text_tensor}
    outputs = model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # Extract image-to-text similarity logits

    # Backward pass to compute gradients
    logits_per_image[:, 0].backward()

    # Ensure hooks are removed after processing
    forward_hook_handle.remove()
    backward_hook_handle.remove()

    return activations, gradients


def apply_gradcam(model, processor, image, text):
    """
    Applies Grad-CAM to visualize important regions used by CLIP for image-text similarity.

    Args:
        model: Pretrained CLIP model.
        processor: Processor for text and images.
        image: PIL Image.
        text: Text input.

    Returns:
        PIL Image with Grad-CAM overlay.
    """
    # Preprocess input
    inputs = processor(images=image, text=[text], return_tensors="pt", padding=True)
    image_tensor = inputs["pixel_values"]
    text_tensor = inputs["input_ids"]

    # Get activations and gradients
    activations, gradients = get_gradcam_activations(model, image_tensor, text_tensor)

    # Compute Grad-CAM
    weights = gradients.mean(
        dim=(2, 3), keepdim=True
    )  # Global-average-pooling for gradients
    gradcam = torch.sum(weights * activations, dim=1).squeeze()
    gradcam = gradcam.relu()  # Apply ReLU
    gradcam = (gradcam - gradcam.min()) / (
        gradcam.max() - gradcam.min()
    )  # Normalize to [0, 1]

    # Convert Grad-CAM to a heatmap
    gradcam_np = gradcam.cpu().detach().numpy()
    gradcam_resized = np.uint8(gradcam_np * 255)
    gradcam_resized = Image.fromarray(gradcam_resized).resize(
        image.size, resample=Image.BICUBIC
    )

    # Overlay Grad-CAM heatmap onto image
    heatmap = np.array(gradcam_resized, dtype=np.float32)
    heatmap = (heatmap - heatmap.min()) / (
        heatmap.max() - heatmap.min()
    )  # Normalize heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).convert("RGB")

    # Blend original image with Grad-CAM overlay
    overlay = Image.blend(image, heatmap, alpha=0.5)

    return overlay


# Function to compare two images with CLIP models and apply Grad-CAM
def compare_clip_models_with_gradcam(source_image, target_image):
    source = Image.open(source_image).convert("RGB")
    target = Image.open(target_image).convert("RGB")

    # Process with Model 1
    inputs_1 = processor_1(
        text=["This is a source image", "This is a target image"],
        images=[source, target],
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs_1 = model_1(**inputs_1)
        similarity_1 = torch.nn.functional.cosine_similarity(
            outputs_1.image_embeds, outputs_1.text_embeds
        )

    # Process with Model 2
    inputs_2 = processor_2(
        text=["This is a source image", "This is a target image"],
        images=[source, target],
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs_2 = model_2(**inputs_2)
        similarity_2 = torch.nn.functional.cosine_similarity(
            outputs_2.image_embeds, outputs_2.text_embeds
        )

    # Apply Grad-CAM
    gradcam_source_1 = apply_gradcam(
        model_1, processor_1, source, "This is a source image"
    )
    gradcam_target_1 = apply_gradcam(
        model_1, processor_1, target, "This is a target image"
    )

    gradcam_source_2 = apply_gradcam(
        model_2, processor_2, source, "This is a source image"
    )
    gradcam_target_2 = apply_gradcam(
        model_2, processor_2, target, "This is a target image"
    )

    # Prepare similarity scores
    result_1 = {
        "Model (Vit-Base-Patch16)": float(similarity_1[0].item()),
        "Model (Vit-Large-Patch14)": float(similarity_2[0].item()),
    }

    return (
        gradcam_source_1,
        gradcam_target_1,
        gradcam_source_2,
        gradcam_target_2,
        result_1,
        "Comparison complete!",
    )


def interface():
    # Gradio Interface
    iface = gr.Interface(
        fn=compare_clip_models_with_gradcam,
        inputs=[
            gr.Image(type="filepath", label="Source Image"),
            gr.Image(type="filepath", label="Target Image"),
        ],
        outputs=[
            gr.Image(label="Grad-CAM Source Image (Model 1)"),
            gr.Image(label="Grad-CAM Target Image (Model 1)"),
            gr.Image(label="Grad-CAM Source Image (Model 2)"),
            gr.Image(label="Grad-CAM Target Image (Model 2)"),
            gr.Label(label="Similarity Scores"),
            gr.Textbox(label="Comparison Summary"),
        ],
    )

    iface.launch()


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.suppress_errors = False

    interface()
