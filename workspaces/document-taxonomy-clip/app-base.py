import os
from time import time
from typing import Dict, List, Tuple

import gradio as gr
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
    "/home/gbugaj/dev/grapnel-tooling/train-clip/oputput-taxonomy/checkpoint-4000"
)

# Load CLIP models and their respective processors
model_1 = CLIPModel.from_pretrained(model1_name)
processor_1 = CLIPProcessor.from_pretrained(model1_name)

model_2 = CLIPModel.from_pretrained(model2_name)
processor_2 = CLIPProcessor.from_pretrained(model2_name)


def compare_clip_models(source_image, target_image):
    # Load images
    source = Image.open(source_image).convert("RGB")
    target = Image.open(target_image).convert("RGB")
    text = "taxonomy of codes"
    # Process images using the first model's processor
    inputs_1 = processor_1(
        text=[text, text], images=[source, target], return_tensors="pt", padding=True
    )

    with torch.no_grad():
        # Get logits for text and images for model 1
        outputs_1 = model_1(**inputs_1)
        image_embeds_1 = outputs_1.image_embeds
        text_embeds_1 = outputs_1.text_embeds

        # Compute similarity for model 1
        similarity_1 = torch.nn.functional.cosine_similarity(
            image_embeds_1, text_embeds_1
        )

    # Process images using the second model's processor
    inputs_2 = processor_2(
        text=[text, text], images=[source, target], return_tensors="pt", padding=True
    )

    with torch.no_grad():
        # Get logits for text and images for model 2
        outputs_2 = model_2(**inputs_2)
        image_embeds_2 = outputs_2.image_embeds
        text_embeds_2 = outputs_2.text_embeds

        # Compute similarity for model 2
        similarity_2 = torch.nn.functional.cosine_similarity(
            image_embeds_2, text_embeds_2
        )

    result_1 = {
        f"Model ({model1_name})": float(similarity_1[0].item()),
        f"Model ({model2_name})": float(similarity_2[0].item()),
    }

    return result_1, "Comparison complete!"


def interface():
    iface = gr.Interface(
        fn=compare_clip_models,
        inputs=[
            gr.Image(type="filepath", label="Source Image"),
            gr.Image(type="filepath", label="Target Image"),
        ],
        outputs=[
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
