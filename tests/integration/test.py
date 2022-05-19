import os

import cv2
from marie.boxes import BoxProcessorCraft
from marie.document import CraftIcrProcessor
from marie.document import TrOcrIcrProcessor
from marie.timer import Timer
from marie.utils.utils import current_milli_time, ensure_exists

# @Timer(text="Creating zip in {:.2f} seconds")
if __name__ == "__main__":
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    img_path = "../../examples/set-001/test/fragment-001.png"

    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    key = img_path.split("/")[-1]
    snippet = cv2.imread(img_path)

    box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="../../model_zoo/craft", cuda=False)
    # icr = CraftIcrProcessor(work_dir=work_dir_icr, models_dir="../../model_zoo/icr", cuda=False)
    icr = TrOcrIcrProcessor(work_dir=work_dir_icr, models_dir="../../model_zoo/trocr", cuda=False)

    boxes, img_fragments, lines, _ = box.extract_bounding_boxes(key, "field", snippet)
    for xx in range(0, 10):
        results = icr.recognize(key, "test", snippet, boxes, img_fragments, lines)
        print(xx)