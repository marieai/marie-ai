import os

import cv2
import torch

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.utils.docs import frames_from_file
from marie.utils.image_utils import crop_to_content
from marie.utils.utils import ensure_exists


def process_image(image):
    box = BoxProcessorUlimDit(
        models_dir="../../model_zoo/unilm/dit/text_detection",
        cuda=True,
    )
    (boxes, fragments, lines, _, lines_bboxes,) = box.extract_bounding_boxes(
        "gradio",
        "field",
        image,
        PSMode.SPARSE,
        bbox_optimization=True,
        bbox_context_aware=True,
    )

    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(image, lines_bboxes, format="xywh")

    bboxes_img.save("/tmp/boxes/bboxes_img.png")
    lines_img.save("/tmp/boxes/lines.png")


if __name__ == "__main__":

    torch.backends.cuda.matmul.allow_tf32 = True

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "~/tmp/PID_576_7188_0_150300411_4.tif"
    # img_path = "~/tmp/demo/merged-001_00001.png"
    img_path = "~/tmp/blowsup-memory/194398480-extracted/194398480-0005.png"
    img_path = "~/tmp/blowsup-memory/194398480-extracted/194398480-0003.png"

    img_path = os.path.expanduser(img_path)
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    key = img_path.split("/")[-1]
    frames = frames_from_file(img_path)
    # scale image up by 2x
    # frames[0] = cv2.resize(frames[0], None, fx=3, fy=3)

    cv2.imwrite(f"/tmp/boxes/resized.png", frames[0])
    frames = [frames[0]]
    # frames = [crop_to_content(frame, True) for frame in frames]

    process_image(frames[0])
