import base64
import urllib.request
from io import BytesIO

import torch
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Initialize the model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Grab a sample PDF
urllib.request.urlretrieve("https://molmo.allenai.org/paper.pdf", "./paper.pdf")

# Render page 1 to an image
image_base64 = render_pdf_to_base64png("./paper.pdf", 1, target_longest_image_dim=1024)

# Build the prompt, using document metadata
anchor_text = get_anchor_text(
    "./paper.pdf", 1, pdf_engine="pdfreport", target_length=4000
)
print("Anchor text:", anchor_text)
prompt = build_finetuning_prompt(anchor_text)

print("Prompt:", prompt)
# Build the full prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            },
        ],
    }
]

# Apply the chat template and processor
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

inputs = processor(
    text=[text],
    images=[main_image],
    padding=True,
    return_tensors="pt",
)
inputs = {key: value.to(device) for (key, value) in inputs.items()}

# Generate the output
output = model.generate(
    **inputs,
    temperature=0.8,
    max_new_tokens=2048,
    num_return_sequences=1,
    do_sample=True,
)

# Decode the output
prompt_length = inputs["input_ids"].shape[1]
new_tokens = output[:, prompt_length:]
text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

print(text_output)
# ['{"primary_language":"en","is_rotation_valid":true,"rotation_correction":0,"is_table":false,"is_diagram":false,"natural_text":"Molmo and PixMo:\\nOpen Weights and Open Data\\nfor State-of-the']
