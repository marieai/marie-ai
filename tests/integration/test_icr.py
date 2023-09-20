import os
import time

import cv2

from marie.boxes import BoxProcessorCraft
from marie.document import CraftOcrProcessor, TrOcrProcessor
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
    snippet = cv2.imread(img_path)

    # print(RegistryHolder.REGISTRY[TrOcrIcrProcessor.__name__](work_dir=work_dir_icr, models_dir="../../model_zoo/trocr", cuda=True))
    # print(RegistryHolder.REGISTRY[TrOcrIcrProcessor.__name__](work_dir=work_dir_icr, models_dir="../../model_zoo/trocr", cuda=True))
    icr = TrOcrProcessor(
        work_dir=work_dir_icr, models_dir="../../model_zoo/trocr", cuda=True
    )

    if True:
        box = BoxProcessorCraft(
            work_dir=work_dir_boxes, models_dir="../../model_zoo/craft", cuda=True
        )

        boxes, img_fragments, lines, _ = box.extract_bounding_boxes(
            key, "field", snippet
        )

        start = time.time()
        for xx in range(0, 1):
            start_iter = time.time()
            results = icr.recognize(key, "test", snippet, boxes, img_fragments, lines)
            print(len(results))
            logger.info("iter time: %s" % (time.time() - start_iter))

        logger.info("Elapsed: %s" % (time.time() - start))
