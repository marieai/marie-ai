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

    img_path = "~/tmp/163611436.tif"
    img_path = "~/tmp/wrong-ocr/169118830.tif"
    img_path = "~/tmp/wrong-ocr/regions/overlay_image_1_9359800610.png"
    img_path = "~/tmp/wrong-ocr/regions/overlay_image_1_9308042272.png"
    img_path = "../../assets/psm/block/block-003.png"
    img_path = "/home/gbugaj/burst/150459314_3_cleaned.tiff"
    img_path = "/home/gbugaj/tmp/analysis/PID_1956_9362_0_177978797/300DPI/PID_1956_9362_0_177978797.tif"
    img_path = "/home/gbugaj/tmp/PID_886_7652_0_157518994.tif"
    img_path = "/home/gbugaj/tmp/analysis/PID_893_7663_0_178966520/burst-clean/PID_893_7663_0_178966520.tif-0005.tif"  # BAD BOXES
    img_path = "~/tmp/analysis/PID_1021_7818_0_180097358/burst/PID_1021_7818_0_180097358-0008.tif"
    # img_path = "/home/gbugaj/tmp/analysis/PID_1956_9362_0_ 177978797/300DPI/frames/PID_1956_9362_0_177978797-0002.tif"
    # img_path = "/home/gbugaj/tmp/analysis/PID_1956_9362_0_177978797/300DPI/frames/clip-001.png"
    img_path = "~/tmp/analysis/DEVOPSSD-54421/178443716.tif"

    img_path = os.path.expanduser(img_path)
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    key = img_path.split("/")[-1]
    frames = frames_from_file(img_path)
    frames = [frames[1]]
    # frames = [crop_to_content(frame, True) for frame in frames]

    process_image(frames[0])
