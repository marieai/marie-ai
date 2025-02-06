import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

# Load LayoutLMv3 processor and model
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=2
)  # Adjust num_labels based on entity labels


def preprocess_document(image_path, ocr_results):
    words = [item['text'] for item in ocr_results]
    boxes = [item['bounding_box'] for item in ocr_results]

    encoding = processor(
        image=image_path,
        text=words,
        boxes=boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    return encoding, words, boxes


# Example OCR output
ocr_results = [
    {"text": "Claim", "bounding_box": [10, 10, 50, 30]},
    {"text": "Number:", "bounding_box": [60, 10, 100, 30]},
    {"text": "MRRVFHAG", "bounding_box": [110, 10, 200, 30]},
]

# Preprocess document
image_path = "path/to/document.png"
inputs, words, boxes = preprocess_document(image_path, ocr_results)

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs)

# Extract entity predictions
predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
entities = [
    (word, box) for word, box, pred in zip(words, boxes, predictions) if pred == 1
]  # Assuming 1 is the "entity" label
print("Extracted Entities:", entities)
