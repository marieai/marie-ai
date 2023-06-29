import os

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

    (
        boxes,
        fragments,
        lines,
        _,
        lines_bboxes,
    ) = box.extract_bounding_boxes("gradio", "field", image, PSMode.SPARSE)

    bboxes_img = visualize_bboxes(image, boxes, format="xywh")
    lines_img = visualize_bboxes(image, lines_bboxes, format="xywh")

    bboxes_img.save("/tmp/boxes/bboxes_img.png")
    lines_img.save("/tmp/boxes/lines.png")


if __name__ == "__main__":
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "~/tmp/analysis/DEVOPSSD-54421/178443716.tif"
    img_path = "~/tmp/PID_576_7188_0_150300411_4.tif"

    img_path = os.path.expanduser(img_path)
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    key = img_path.split("/")[-1]
    frames = frames_from_file(img_path)
    frames = [frames[0]]
    # frames = [crop_to_content(frame, True) for frame in frames]

    process_image(frames[0])
