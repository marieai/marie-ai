import torch
from PIL import Image
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

# Load LayoutLMv3 processor and model
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=2
)  # Adjust num_labels based on entity labels


def preprocess_document(image, ocr_results):
    words = [item['text'] for item in ocr_results]
    boxes = [item['bounding_box'] for item in ocr_results]

    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    return encoding, words, boxes


def extract_relations(text):
    """
    Extract relations from text using REBEL.
    """
    # Load REBEL model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)
        print(outputs)

    # Decode predictions
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_text


def extract_key_value_pairs(image_path, ocr_results):
    """
    Complete pipeline for extracting entities and relations from a document.
    """

    # Step 1: Entity Extraction with LayoutLMv3
    inputs, words, boxes = preprocess_document(image_path, ocr_results)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    entities = [
        (word, box) for word, box, pred in zip(words, boxes, predictions) if pred == 1
    ]

    # Step 2: Relation Extraction with REBEL
    entity_text = " ".join(
        [entity[0] for entity in entities]
    )  # Combine entities into a single text input
    relations = extract_relations(entity_text)

    return {"entities": entities, "relations": relations}


# Example OCR results
ocr_results = [
    {"text": "Claim", "bounding_box": [10, 10, 50, 30]},
    {"text": "Number:", "bounding_box": [60, 10, 100, 30]},
    {"text": "MRRVFHAG", "bounding_box": [110, 10, 200, 30]},
    {"text": "Provider", "bounding_box": [10, 50, 50, 70]},
    {"text": "RENUKA", "bounding_box": [60, 50, 100, 70]},
    {"text": "MOPURU", "bounding_box": [110, 50, 200, 70]},
]


# Load the image
image_path = "0011973451.png"
image = Image.open(image_path).convert("RGB")
result = extract_key_value_pairs(image, ocr_results)

# Output results
print("Entities:", result["entities"])
print("Relations:", result["relations"])
