import os

import cv2
import numpy as np
import tqdm

from marie.boxes.box_processor import PSMode
from marie.document import TrOcrIcrProcessor
from marie.executor.text_extraction_executor import CoordinateFormat
from marie.renderer import PdfRenderer
from marie.renderer.text_renderer import TextRenderer
from marie.utils.json import store_json_object
from marie.utils.utils import ensure_exists

if True:
    from marie.boxes import BoxProcessorUlimDit
    from marie.boxes.craft_box_processor import BoxProcessorCraft
    from marie.boxes.textfusenet_box_processor import BoxProcessorTextFuseNet
    from marie.document.craft_icr_processor import CraftIcrProcessor


if __name__ == "__main__":
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "/home/gbugaj/tmp/marie-cleaner/161970410/burst/PID_1956_9362_0_161970410_page_0004.tif"
    # img_path = "/home/greg/tmp/PID_576_7188_0_150300411_4.tif"

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    if True:
        key = img_path.split("/")[-1]
        image = cv2.imread(img_path)

        box = BoxProcessorUlimDit(
            work_dir=work_dir_boxes,
            models_dir="./model_zoo/unilm/dit/text_detection",
            cuda=True,
        )

        # box = BoxProcessorCraft(work_dir=work_dir_boxes, cuda=True)

        (
            boxes,
            fragments,
            lines,
            _,
            lines_bboxes,
        ) = box.extract_bounding_boxes(key, "field", image, PSMode.SPARSE)

        if True:
            # icr = CraftIcrProcessor(work_dir=work_dir_icr, cuda=False)
            icr = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=True)

            result, overlay_image = icr.recognize(key, "test", image, boxes, fragments, lines)

            # need to decorate our results META data
            result["meta"]["page"] = 0
            result["meta"]["lines"] = lines
            result["meta"]["lines_bboxes"] = lines_bboxes
            result["meta"]["format"] = CoordinateFormat.XYWH.name.lower()

            results = [result]
            print("Testing text render")
            cv2.imwrite("/tmp/fragments/overlay.png", overlay_image)

            store_json_object(results, os.path.join("/tmp/fragments", "results.json"))

            if True:
                renderer = PdfRenderer(config={"preserve_interword_spaces": True})
                renderer.render(
                    [image],
                    results,
                    output_filename=os.path.join(work_dir_icr, "results.pdf"),
                )

            renderer = TextRenderer(config={"preserve_interword_spaces": True})
            renderer.render(
                [image],
                results,
                output_filename=os.path.join(work_dir_icr, "results.txt"),
            )
