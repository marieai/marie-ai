import os
import time

import cv2

from marie.boxes import BoxProcessorUlimDit
from marie.document import TrOcrProcessor
from marie.document.layoutreader import TextLayout
from marie.executor.ner.utils import normalize_bbox
from marie.logger import setup_logger
from marie.registry_base import RegistryHolder
from marie.timer import Timer
from marie.utils.json import load_json_file, store_json_object
from marie.utils.utils import current_milli_time, ensure_exists

logger = setup_logger(__name__)

# @Timer(text="Creating zip in {:.2f} seconds")
if __name__ == "__main__":
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    img_path = "../../examples/set-001/test/fragment-001.png"
    img_path = "~/tmp/layoutreader/fragment-004.png"

    img_path = os.path.expanduser(img_path)
    file_name = os.path.basename(img_path).split(".")[0]

    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    key = img_path.split("/")[-1]
    image = cv2.imread(img_path)

    text_layout = TextLayout("../../model_zoo/unilm/layoutreader/layoutreader-base-readingbank")
    run_ocr = False

    if run_ocr:
        icr = TrOcrProcessor(
            work_dir=work_dir_icr, models_dir="../../model_zoo/trocr", cuda=True
        )

        box = BoxProcessorUlimDit(
            work_dir=work_dir_boxes, models_dir="../../model_zoo/craft", cuda=True
        )

        (
            boxes,
            fragments,
            lines,
            _,
            lines_bboxes,
        ) = box.extract_bounding_boxes(
            key, "field", image
        )

    start = time.time()
    for xx in range(0, 1):
        start_iter = time.time()

        if run_ocr:
            results = icr.recognize(key, "test", image, boxes, fragments, lines)
            print(len(results))
            logger.info("iter time: %s" % (time.time() - start_iter))

            result = results[0]
            print(result)
            # get boxes and words from the result
            words = []
            boxes = []
            # boxes = [word["box"] for word in words]
            for word in result["words"]:
                x, y, w, h = word["box"]
                # convert to x1, y1, x2, y2
                w_box = [x, y, x + w, y + h]
                word["box"] = w_box
                words.append(word["text"])
                boxes.append(normalize_bbox(word["box"], (image.shape[1], image.shape[0])))

            stored = {
                "words": words,
                "boxes": boxes,
            }

            store_json_object(stored, f"~/tmp/layoutreader/{file_name}.json")

        stored = load_json_file(f"~/tmp/layoutreader/{file_name}.json")

        words = stored["words"]
        boxes = stored["boxes"]

        print("len words", len(words))
        print("len boxes", len(boxes))

        tl_words, tl_boxes = text_layout(words, boxes)

        print(tl_words)
        print(tl_boxes)

    logger.info("Elapsed: %s" % (time.time() - start))
