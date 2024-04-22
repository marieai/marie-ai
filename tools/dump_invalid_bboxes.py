import glob
import os

import torch as torch
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.document import TrOcrProcessor
from marie.utils.ocr_debug import dump_bboxes

use_cuda = torch.cuda.is_available()


def build_ocr_engine():
    box_processor = BoxProcessorUlimDit(
        models_dir="/mnt/data/marie-ai/model_zoo/unilm/dit/text_detection",
        cuda=use_cuda,
    )

    icr_processor = TrOcrProcessor(
        models_dir="/mnt/data/marie-ai/model_zoo/trocr", cuda=use_cuda
    )

    return box_processor, icr_processor


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

    dump_bboxes(image, result, prefix=name, threshold=0.95)


def process_dir(image_dir: str):
    box_processor, icr_processor = build_ocr_engine()
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
        try:
            print(img_path)
            process_image(img_path, box_processor, icr_processor)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    process_dir(
        "/home/gbugaj/tmp/analysis/BAD-OCR-IN-GRAPNEL/PID_4730_11946_0_201758379"
    )
