import os

import numpy as np
import tqdm

import cv2
from marie.renderer.text_renderer import TextRenderer
from marie.boxes.box_processor import PSMode
from marie.utils.utils import ensure_exists


if True:
    from marie.boxes.craft_box_processor import BoxProcessorCraft
    from marie.boxes.textfusenet_box_processor import BoxProcessorTextFuseNet
    from marie.document.craft_icr_processor import CraftIcrProcessor


if __name__ == "__main__":

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "./assets/psm/word/0001.png"
    img_path = "./assets/english/Scanned_documents/Picture_029.tif"
    # img_path = './assets/english/Scanned_documents/t2.tif'
    img_path = "./assets/english/Scanned_documents/Picture_010.tif"
    img_path = "./assets/english/Lines/002.png"
    # img_path = './assets/english/Lines/001.png'
    # img_path = './assets/english/Lines/003.png'
    # img_path = './assets/english/Lines/005.png'
    # img_path = './assets/english/Lines/004.png'
    img_path = "./assets/private/PID_576_7188_0_149495857_page_0002.tif"
    # img_path = "/home/gbugaj/data/private/coco-text/000005.tif"
    img_path = "/home/gbugaj/tmp/marie-cleaner/161970410/burst/PID_1956_9362_0_161970410_page_0004.tif"

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    if True:
        key = img_path.split("/")[-1]
        image = cv2.imread(img_path)
        mean, std = cv2.meanStdDev(image)

        # box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir='./model_zoo/craft', cuda=True)
        box = BoxProcessorTextFuseNet(
            work_dir=work_dir_boxes, models_dir="./models/fusenet", cuda=False
        )
        boxes, img_fragments, lines, _ = box.extract_bounding_boxes(
            key, "field", image, PSMode.SPARSE
        )

        if True:
            icr = CraftIcrProcessor(work_dir=work_dir_icr, cuda=False)
            result, overlay_image = icr.recognize(
                key, "test", image, boxes, img_fragments, lines
            )

            print("Testing text render")

            renderer = TextRenderer(config={"preserve_interword_spaces": True})
            renderer.render(image, result)
