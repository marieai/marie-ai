import glob
import os

import torch as torch
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.constants import __model_path__
from marie.document import TrOcrProcessor

use_cuda = torch.cuda.is_available()


def build_ocr_engines():
    # return None, None, None

    box_processor = BoxProcessorUlimDit(
        models_dir="/mnt/data/marie-ai/model_zoo/unilm/dit/text_detection",
        cuda=use_cuda,
    )

    ocr2_processor = TrOcrProcessor(
        # model_name_or_path=os.path.join(__model_path__, "trocr", "tuned", "checkpoint_best.pt"),
        model_name_or_path=os.path.join(
            __model_path__, "trocr", "trocr-large-printed.pt"
        ),
        cuda=use_cuda,
    )

    ocr1_processor = TrOcrProcessor(
        model_name_or_path=os.path.join(
            __model_path__, "trocr", "trocr-large-printed.pt"
        ),
        cuda=use_cuda,
    )

    return box_processor, ocr1_processor, ocr2_processor


def process_image(img_path, box_processor, ocr1_processor, ocr2_processor):
    image = Image.open(img_path).convert("RGB")
    name = os.path.basename(img_path)
    name = os.path.splitext(name)[0]
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    (
        boxes,
        fragments,
        lines,
        _,
        lines_bboxes,
    ) = box_processor.extract_bounding_boxes("gradio", "field", image, PSMode.SPARSE)

    result1, overlay_image1 = ocr1_processor.recognize(
        "gradio ", "00000", image, boxes, fragments, lines, return_overlay=True
    )

    result2, overlay_image2 = ocr2_processor.recognize(
        "gradio ", "00000", image, boxes, fragments, lines, return_overlay=True
    )

    # iterate over both results and save them
    for word1, word2 in zip(result1["words"], result2["words"]):
        print("word : ", word1, word2)

        if word1["text"] != word2["text"]:
            print("DIFFERENT")
            print(word1)
            print(word2)
            print("----")


def process_dir(image_dir: str, box_processor, ocr1_processor, ocr2_processor):
    import random

    items = glob.glob(os.path.join(image_dir, "*.*"))
    random.shuffle(items)

    for idx, img_path in enumerate(items):
        try:
            print(img_path)
            process_image(img_path, box_processor, ocr1_processor, ocr2_processor)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    box_processor, ocr1_processor, ocr2_processor = build_ocr_engines()

    process_dir(
        "/home/gbugaj/tmp/analysis/BAD-OCR-IN-GRAPNEL/202585877/XXX",
        box_processor,
        ocr1_processor,
        ocr2_processor,
    )
