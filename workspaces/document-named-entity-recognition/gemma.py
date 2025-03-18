# pip install accelerate

import base64

import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

model_id = "google/gemma-3-12b-it"


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit loading
    llm_int8_threshold=6.0,  # Threshold for mixed-precision
)

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation='flash_attention_2',
    quantization_config=bnb_config,
).eval()

processor = AutoProcessor.from_pretrained(model_id)


#  base 64 encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_path = "/home/gbugaj/dev/workflow/grapnel-g5/assets/TID-100985/ANNOTATED/OUT/tables/fragments/30_1.png"
base64_image = encode_image(image_path)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
            # {"type": "image", "image": "http://0.0.0.0:8088/10_0.png"},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    },
]


inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=250, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
# focusing on a cluster of pink cosmos flowers and a busy bumblebee.
# It has a slightly soft, natural feel, likely captured in daylight.
