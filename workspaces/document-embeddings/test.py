from typing import List, Tuple

from PIL import Image
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor
from util import process_image

from marie.boxes import BoxProcessorUlimDit
from marie.document import TrOcrProcessor
from marie.executor.ner.utils import normalize_bbox
from marie.ocr.util import get_words_and_boxes
from marie.utils.docs import convert_frames


def build_ocr_engine():
    text_layout = None
    box_processor = BoxProcessorUlimDit(
        models_dir="/mnt/data/marie-ai/model_zoo/unilm/dit/text_detection",
        cuda=True,
    )

    icr_processor = TrOcrProcessor(
        models_dir="/mnt/data/marie-ai/model_zoo/trocr", cuda=True
    )

    return box_processor, icr_processor, text_layout


def preprocess(
    frames: List, words: List[List[str]], boxes: List[List[List[int]]]
) -> Tuple[List, List[List[str]], List[List[List[int]]]]:
    """Preprocess the input data for inference. This method is called by the predict method.
    :param frames: The frames to be preprocessed.
    :param words: The words to be preprocessed.
    :param boxes: The boxes to be preprocessed, in the format (x, y, w, h).
    :returns: The preprocessed frames, words, and boxes (normalized).
    """
    assert len(frames) == len(boxes) == len(words)
    frames = convert_frames(frames, img_format="pil")
    normalized_boxes = []

    for frame, box_set, word_set in zip(frames, boxes, words):
        if not isinstance(frame, Image.Image):
            raise ValueError("Frame should have been an PIL.Image instance")
        nbox = []
        for i, box in enumerate(box_set):
            nbox.append(normalize_bbox(box_set[i], (frame.size[0], frame.size[1])))
        normalized_boxes.append(nbox)

    assert len(frames) == len(normalized_boxes) == len(words)

    return frames, words, normalized_boxes


# Load model and processor
model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
)

# Load image and process
image = Image.open("/home/greg/dev/flan-t5-text-classifier/sample-001.png")
if image.mode != "RGB":
    image = image.convert("RGB")

box_processor, ocr_processor, _ = build_ocr_engine()
bboxes_img, overlay_image, lines_img, result_json, extracted_text = process_image(
    image, box_processor, ocr_processor
)
words, boxes = get_words_and_boxes(result_json, 0)
print('-' * 50)

print(f"Words : {len(words)} ::  {words}")
print(f"Boxes : {len(boxes)}")
assert len(words) == len(boxes)

frames, words, normalized_boxes = preprocess([image], [words], [boxes])

image = frames[0]
words = words[0]
boxes = normalized_boxes[0]

inputs = processor(
    # fmt: off
    image,
    words,
    boxes=boxes,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
    max_length=512,
    # fmt: on
)

# inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)
print(outputs)
document_embedding = outputs.logits.detach().numpy()
print(document_embedding.shape)
print(document_embedding)
