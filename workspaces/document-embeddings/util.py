import re
from typing import Union

import numpy as np
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

    return bboxes_img, overlay_image, lines_img, result, to_text(result)
