import os

import numpy as np
import tqdm

import cv2

from marie.document import TrOcrIcrProcessor
from marie.renderer.text_renderer import TextRenderer
from marie.boxes.box_processor import PSMode
from marie.utils.utils import ensure_exists


if True:
    from marie.boxes.craft_box_processor import BoxProcessorCraft
    from marie.boxes.textfusenet_box_processor import BoxProcessorTextFuseNet
    from marie.boxes import BoxProcessorUlimDit
    from marie.document.craft_icr_processor import CraftIcrProcessor


if __name__ == "__main__":

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "/home/gbugaj/tmp/marie-cleaner/161970410/burst/PID_1956_9362_0_161970410_page_0004.tif"

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    if True:
        key = img_path.split("/")[-1]
        image = cv2.imread(img_path)
        mean, std = cv2.meanStdDev(image)

        # box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir='./model_zoo/craft', cuda=True)
        box = BoxProcessorUlimDit(
            work_dir=work_dir_boxes,
            models_dir="./model_zoo/unilm/dit/text_detection",
            cuda=True,
        )
        (boxes, fragments, lines, _, lines_bboxes,) = box.extract_bounding_boxes(
            key, "field", image, PSMode.SPARSE
        )

        if True:
            # icr = CraftIcrProcessor(work_dir=work_dir_icr, cuda=False)
            icr = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=True)

            result, overlay_image = icr.recognize(
                key, "test", image, boxes, fragments, lines
            )

            print("Testing text render")

            # box = BoxProcessorTextFuseNet(work_dir=work_dir_boxes, models_dir='./models/fusenet', cuda=False)
            # icr = CraftIcrProcessor(work_dir=work_dir_icr, cuda=False)

            cv2.imwrite("/tmp/fragments/overlay.png", overlay_image)
            print(result)
            json_path = os.path.join("/tmp/fragments", "results.json")

            renderer = TextRenderer(config={"preserve_interword_spaces": True})
            renderer.render(
                image, result, output_filename=os.path.join(work_dir_icr, "results.txt")
            )
