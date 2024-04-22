import glob
import os
import uuid

import cv2
import torch as torch
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.document import CraftOcrProcessor, TrOcrProcessor
from marie.utils.ocr_debug import dump_bboxes, normalize_label
from marie.utils.utils import ensure_exists

use_cuda = torch.cuda.is_available()


def build_ocr_engines():
    # return None, None, None

    box_processor = BoxProcessorUlimDit(
        models_dir="/mnt/data/marie-ai/model_zoo/unilm/dit/text_detection",
        cuda=use_cuda,
    )

    trocr_processor = TrOcrProcessor(
        models_dir="/mnt/data/marie-ai/model_zoo/trocr", cuda=use_cuda
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

    dump_bboxes(image, result, prefix=name, threshold=0.90)


def process_dir(image_dir: str):
    box_processor, trocr_processor, craf_processor = build_ocr_engines()
    engines = {"trocr": trocr_processor, "craft": craf_processor}

    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
        try:
            print(img_path)
            process_image(img_path, box_processor, trocr_processor)
        except Exception as e:
            print(e)


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
    engines = {"trocr": trocr_processor, "craft": craf_processor}

    fragment = cv2.imread(
        "/home/greg/datasets/SROIE_OCR/lines/icr/162305841_6_1713596222984/lines/162305841_6_38_0.9737.png"
    )
    result = trocr_processor.recognize_from_fragments([fragment])

    print("Result: ", result)

    # Before:  {'confidence': 0.6603, 'id': 'img-0', 'text': 'INN3802 409710 00 THEPPENTO BEROSS'}
    # AFTER    {'confidence': 0.9704, 'id': 'img-0', 'text': '07/12/2022 430 97110 GO THERAPEUTIC EXERCISES 2 $398.50 $91.60'}]

    if False:
        process_dir(
            # "/home/greg/datasets/funsd_dit/IMAGES/LbxIDImages_boundingBox_6292023", --DONE
            # "/home/greg/datasets/funsd_dit/IMAGES/bboxes/03-2024", -- DONE
            # "/home/greg/datasets/corr-indexer/traindeck-raw-01/images/corr-indexing/train",  # -- DONE
            # "/home/greg/datasets/private/eob-extract/converted/imgs/eob-extract/eob-002",
            "/home/greg/datasets/private/medical_page_classification/raw_v2/EOB",
            box_processor,
            trocr_processor,
        )

    # verify_dir("/tmp/boxes/alpha", craf_processor)
