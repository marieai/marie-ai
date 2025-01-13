import base64
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from io import BytesIO
from typing import Union

import numpy as np
import openai
from dotenv import load_dotenv
from PIL import Image

from marie.boxes import PSMode
from marie.boxes.box_processor import BoxProcessor
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.document.ocr_processor import OcrProcessor
from marie.executor.ner.utils import normalize_bbox
from marie.ocr.util import meta_to_text
from marie.utils.json import to_json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"


def save_to_json_file(
    results: dict,
    input_image: Union[np.ndarray, Image.Image],
    output_file="output.json",
):
    """
    Save the prompt, results, and input image (PIL format) to a JSON file.

    Args:
        results (str | dict): The returned results from the model (can be a string or dictionary).
        input_image (PIL.Image.Image): The input image in PIL format.
        output_file (str): The name of the JSON output file.
    """
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    try:
        buffer = BytesIO()
        input_image.save(
            buffer, format="PNG"
        )  # Convert to PNG format (you can choose others)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        encoded_image = None  # Handle conversion error
        print(f"Failed to process input image: {e}")

    data = {
        "timestamp": datetime.now().isoformat(),
        "input_image": {"format": "PNG", "base64_encoding": encoded_image},
    }
    data.update(results)

    try:
        with open(output_file, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully written to {output_file}.")
    except Exception as e:
        print(f"Failed to write JSON file: {e}")


def generate_unique_key(filename: str) -> str:
    """
    Generate a unique key from a filename.

    Args:
        filename (str): The name of the file being uploaded.

    Returns:
        str: A unique key generated from the filename.
    """
    base_name = os.path.basename(filename)
    filename, file_extension = os.path.splitext(base_name)
    unique_key = f"{filename}_{uuid.uuid4().hex}_{hashlib.sha256(base_name.encode()).hexdigest()[:10]}"
    return unique_key


def to_text(result: dict) -> str:
    """
    Create a text representation of the result from OCR.
    """
    text = meta_to_text([result])
    collapsed_text = re.sub(r'\n\s*\n+', '\n', text)
    return collapsed_text


def process_image(
    image: Union[np.ndarray, Image.Image],
    box_processor: BoxProcessor,
    ocr_processor: OcrProcessor,
):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # image = Image.open(uploaded_file).convert("RGB")
    (
        boxes,
        fragments,
        lines,
        _,
        lines_bboxes,
    ) = box_processor.extract_bounding_boxes("streamlit", "field", image, PSMode.SPARSE)

    result, overlay_image = ocr_processor.recognize(
        "streamlit", "00000", image, boxes, fragments, lines, return_overlay=True
    )

    # Normalize bounding boxes
    words = []
    boxes_norm = []
    for word in result["words"]:
        x, y, w, h = word["box"]
        w_box = [x, y, x + w, y + h]
        words.append(word["text"])
        boxes_norm.append(normalize_bbox(w_box, (image.size[0], image.size[1])))

    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(overlay_image, lines_bboxes, format="xywh")

    return bboxes_img, overlay_image, lines_img, to_json(result), to_text(result)
