import os

import cv2
import torch

from marie.boxes import BoxProcessorCraft, BoxProcessorUlimDit, PSMode
from marie.boxes.dit.ulim_dit_box_processor import visualize_bboxes
from marie.utils.docs import frames_from_file
from marie.utils.image_utils import crop_to_content
from marie.utils.utils import ensure_exists


def process_image(image):
    box = BoxProcessorUlimDit(
        models_dir="../../model_zoo/unilm/dit/text_detection",
        cuda=True,
        refinement=True,
    )

    # box = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir='../../model_zoo/craft', cuda=True)

    (boxes, fragments, lines, _, lines_bboxes,) = box.extract_bounding_boxes(
        "gradio",
        "field",
        image,
        PSMode.SPARSE,
        # bbox_optimization=True,
        # bbox_context_aware=True,
    )

    bboxes_img = visualize_bboxes(image, boxes, format="xywh", blackout=False, blackout_color=(255, 255, 255, 255))
    lines_img = visualize_bboxes(image, lines_bboxes, format="xywh")

    bboxes_img.save("/tmp/boxes/bboxes_img.png")
    lines_img.save("/tmp/boxes/lines.png")


if __name__ == "__main__":

    torch.backends.cuda.matmul.allow_tf32 = True

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")
    img_path = "~tmp/demo/159035444_1.png" # black

    img_path = os.path.expanduser(img_path)
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    key = img_path.split("/")[-1]
    frames = frames_from_file(img_path)
    # frames[0] = cv2.resize(frames[0], None, fx=3, fy=3)

    cv2.imwrite(f"/tmp/boxes/resized.png", frames[0])
    frames = [frames[0]]
    # frames = [crop_to_content(frame, True) for frame in frames]

    process_image(frames[0])

    # BoxProcessorCraft