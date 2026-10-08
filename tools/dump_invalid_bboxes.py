import argparse
import glob
import os
import uuid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directory containing images to process"
    )
    parser.add_argument(
        "image_dir",
        type=os.path.expanduser,
        help="Directory containing images to process",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

import cv2
import torch as torch
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.constants import __model_path__
from marie.document import CraftOcrProcessor, TrOcrProcessor
from marie.utils.ocr_debug import dump_bboxes, normalize_label
from marie.utils.utils import ensure_exists

use_cuda = torch.cuda.is_available()


def build_ocr_engines():
    # return None, None, None

    box_processor = BoxProcessorUlimDit(
        models_dir=os.path.join(__model_path__, "unilm", "dit", "text_detection"),
        cuda=use_cuda,
    )

    trocr_processor = TrOcrProcessor(
        models_dir=os.path.join(__model_path__, "trocr"), cuda=use_cuda
    )

    craft_processor = CraftOcrProcessor(cuda=True)

    return box_processor, trocr_processor, craft_processor


def process_image(img_path, box_processor, icr_processor):
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

    result, overlay_image = icr_processor.recognize(
        "gradio ", "00000", image, boxes, fragments, lines, return_overlay=True
    )

    # text_filters = ["PATIENT:", "NAME:", "MEMBER:"],
    # ngram = 3,
    dump_bboxes(
        image,
        result,
        prefix=name,
        threshold=0.90,
        text_filters=[
            "DATE:",
            "ACCT#:",
            "ACCT:",
            "ACCT",
            "ACCOUNT",
            "CLAIM#:",
            "CLAIM:",
            "NUMBER:",
        ],
        ngram=3,
    )


def process_dir(image_dir: str, box_processor, trocr_processor):
    import random

    items = glob.glob(os.path.join(image_dir, "*.*"))
    random.shuffle(items)

    for idx, img_path in enumerate(items):
        try:
            print(img_path)
            process_image(img_path, box_processor, trocr_processor)
        except Exception as e:
            print(e)
            raise e


def _verify_dir(text_to_validate, image_dir: str, ocr_processor):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.png"))):
        try:
            # image = Image.open(img_path).convert("RGB")
            image = cv2.imread(img_path)
            results = ocr_processor.recognize_from_fragments([image])
            if results:
                if len(results) > 0:
                    result = results[0]
                    text = result["text"]
                    confidence = result["confidence"]
                    print(f"Text: {text}, Confidence: {confidence}")
                    validated = False
                    if text == text_to_validate:
                        validated = True

                    label = normalize_label(text)
                    # check if text is only numbers
                    root_label = f"alpha"
                    if text.isdigit():
                        root_label = f"number"

                    ensure_exists(
                        f"/tmp/boxes/validated-{validated}/{root_label}/{label}"
                    )
                    with open(
                        f"/tmp/boxes/validated-{validated}/{root_label}/{label}/label.txt",
                        "w",
                    ) as f:
                        f.write(text)
                        f.write(f"\n")

                    # create a unique filename to prevent overwriting using uuid
                    fname = uuid.uuid4().hex
                    word_img = Image.open(img_path)
                    original_filename = os.path.basename(img_path)
                    output_path = f"/tmp/boxes/validated-{validated}/{root_label}/{label}/{original_filename}"
                    print(output_path)
                    word_img.save(
                        # f"/tmp/boxes/validated-{validated}/{root_label}/{label}/{idx}_{fname}.png"
                        output_path
                    )

        except Exception as e:
            print(e)
        return False


def verify_dir(image_dir: str, ocr_processor):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "**"))):
        try:
            # read label from label.txt
            with open(os.path.join(img_path, "label.txt"), "r") as f:
                text = f.read().strip()

            print(text)
            print(img_path)
            validated = _verify_dir(text, img_path, ocr_processor)

        except Exception as e:
            raise e


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    box_processor, trocr_processor, craf_processor = build_ocr_engines()
    process_dir(args.image_dir, box_processor, trocr_processor)
