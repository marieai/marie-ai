import os

import torch

from marie.boxes.box_processor import PSMode
from marie.ocr import CoordinateFormat, DefaultOcrEngine
from marie.renderer import PdfRenderer
from marie.renderer.text_renderer import TextRenderer
from marie.utils.docs import frames_from_file
from marie.utils.json import store_json_object, load_json_file
from marie.utils.utils import ensure_exists


if __name__ == "__main__":
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "~/tmp/163611436.tif"
    img_path = os.path.expanduser(img_path)

    print(img_path)
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    key = img_path.split("/")[-1]
    frames = frames_from_file(img_path)

    if True:
        use_cuda = torch.cuda.is_available()
        ocr_engine = DefaultOcrEngine(cuda=use_cuda)
        results = ocr_engine.extract(frames, PSMode.SPARSE, CoordinateFormat.XYWH)

        print(results)
        print("Testing text render")
        store_json_object(results, os.path.join("/tmp/fragments", "results.json"))

    results = load_json_file(os.path.join("/tmp/fragments", "results.json"))

    renderer = PdfRenderer(config={"preserve_interword_spaces": True})
    renderer.render(
        frames,
        results,
        output_filename=os.path.join(work_dir_icr, "results.pdf"),
    )

    renderer = TextRenderer(config={"preserve_interword_spaces": True})
    renderer.render(
        frames,
        results,
        output_filename=os.path.join(work_dir_icr, "results.txt"),
    )
