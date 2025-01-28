from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)


def get_image_embedding(image, text):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    image_embeddings = model.get_image_features(**inputs)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

    inputs = processor(text=text, return_tensors="pt", padding=True)
    text_embeddings = model.get_text_features(**inputs)

    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    combined_embeddings = (image_embeddings + text_embeddings) / 2

    # image_weight = 0.4
    # text_weight = 0.6
    # combined_embeddings = (image_weight * image_embeddings) + (text_weight * text_embeddings)
    return combined_embeddings


# Load image and process
image = Image.open("/home/greg/dev/flan-t5-text-classifier/sample-001.png")
if image.mode != "RGB":
    image = image.convert("RGB")
text = "Patient Name: Robert Ullman, Claim ID: E5JNHRHSW00"

emb1 = get_image_embedding(image, "Patient Name: Robert Ullman, Claim ID: E5JNHRHSW00")
emb2 = get_image_embedding(image, "Patient Name: ")

# Example: similarity between two embeddings
similarity = cosine_similarity(emb1, emb2, dim=0)
similarity_mean = similarity.mean()

print("Cosine Similarity:", similarity_mean.item())

# Normalize similarity to the range 0-1
# 1 (Maximum Similarity):**
# 0 (Minimum Similarity):**
# 0.5 (Neutral Similarity):**
# -1 (Maximum Dissimilarity):**

normalized_similarity = (similarity_mean + 1) / 2
print("Cosine Similarity (normalized):", normalized_similarity.item())
