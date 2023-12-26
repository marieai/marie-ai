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
from marie.utils.utils import current_milli_time, ensure_exists

logger = setup_logger(__name__)

# @Timer(text="Creating zip in {:.2f} seconds")
if __name__ == "__main__":
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    img_path = "../../examples/set-001/test/fragment-001.png"

    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    key = img_path.split("/")[-1]
    image = cv2.imread(img_path)

    text_layout = TextLayout("../../model_zoo/unilm/layoutreader/layoutreader-base-readingbank")

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
            w_box = [x, y, x + w, y + h]
            words.append(word["text"])
            boxes.append(normalize_bbox(word["box"], (image.shape[1], image.shape[0])))

        print("len words", len(words))
        print("len boxes", len(boxes))
        layout_boxes = text_layout.forward(words, boxes)
        print(layout_boxes)


    logger.info("Elapsed: %s" % (time.time() - start))
