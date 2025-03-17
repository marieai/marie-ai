import os.path

import gradio as gr
import torch
from transformers import CLIPModel, CLIPProcessor

# Load two different CLIP models
model1_name = "openai/clip-vit-base-patch16"
model2_name = "openai/clip-vit-large-patch14"

model1_name = "openai/clip-vit-large-patch14-336"
model2_name = "openai/clip-vit-base-patch16"
model2_name = os.path.expanduser(
    "~/dev/grapnel-tooling/train-clip/oputput-taxonomy/checkpoint-13000"
)

# Load CLIP models and their respective processors
model_1 = CLIPModel.from_pretrained(model1_name)
processor_1 = CLIPProcessor.from_pretrained(model1_name)
model_2 = CLIPModel.from_pretrained(model2_name)
processor_2 = CLIPProcessor.from_pretrained(model2_name)


def compute_image_similarity(image1_path, image2_path, model, processor):
    """
    Compute similarity between two images using CLIP.

    Args:
    - image1_path (PIL.Image.Image): First input image.
    - image2_path (PIL.Image.Image): Second input image.

    Returns:
    - similarity_score (float): Cosine similarity score between the two images.
    """
    # Preprocess the images
    inputs = processor(
        images=[image1_path, image2_path], return_tensors="pt", padding=True
    )

    # Generate image embeddings
    with torch.no_grad():
        image_features = model.get_image_features(inputs['pixel_values'])

    # Normalize embeddings to unit vectors
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity between the two images
    similarity_score = torch.nn.functional.cosine_similarity(
        image_features[0:1], image_features[1:2]
    ).item()

    return similarity_score


# Define the Gradio interface function
def gradio_clip_interface(image1, image2):
    """
    Gradio interface for comparing image similarity with CLIP.

    Args:
    - image1 (PIL.Image.Image): First uploaded image.
    - image2 (PIL.Image.Image): Second uploaded image.

    Returns:
    - Comparison summary and similarity score.
    """
    similarity_1 = compute_image_similarity(image1, image2, model_1, processor_1)
    similarity_2 = compute_image_similarity(image1, image2, model_2, processor_2)
    return {
        "Model 1 Similarity": f"{similarity_1:.4f}",
        "Model 2 Similarity": f"{similarity_2:.4f}",
    }


def interface():
    iface = gr.Interface(
        fn=gradio_clip_interface,
        inputs=[
            gr.Image(type="pil", label="Upload First Image"),
            gr.Image(type="pil", label="Upload Second Image"),
        ],
        outputs=gr.JSON(label="Similarity Scores"),
        title="CLIP Image Similarity",
        description="Upload two images to compute their similarity using CLIP.",
    )

    iface.launch()


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    # torch._dynamo.config.suppress_errors = False

    interface()
