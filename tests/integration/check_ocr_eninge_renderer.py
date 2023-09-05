import glob
import os
from typing import Dict

import torch

from marie.boxes.box_processor import PSMode
from marie.ocr import CoordinateFormat, DefaultOcrEngine
from marie.renderer import PdfRenderer
from marie.renderer.text_renderer import TextRenderer
from marie.utils.docs import frames_from_file
from marie.utils.image_utils import crop_to_content
from marie.utils.json import store_json_object, load_json_file
from marie.utils.utils import ensure_exists
from marie.timer import Timer


def process_dir(
        ocr_engine: DefaultOcrEngine,
        image_dir: str,
):
    import random
    items = glob.glob(os.path.join(image_dir, "*.*"))
    random.shuffle(items)

    for idx, img_path in enumerate(items):
        try:
            process_file(ocr_engine, img_path)
        except Exception as e:
            print(e)
            # raise e


@Timer(text="Process time {:.4f} seconds")
def process_file(ocr_engine: DefaultOcrEngine, img_path: str):
    try:
        print("Processing", img_path)
        img_path = os.path.expanduser(img_path)
        if not os.path.exists(img_path):
            raise Exception(f"File not found : {img_path}")

        key = img_path.split("/")[-1]
        frames = frames_from_file(img_path)

        results = ocr_engine.extract(frames, PSMode.SPARSE, CoordinateFormat.XYWH)

        print("Testing text renderer")

        store_json_object(results, os.path.join("/tmp/fragments", f"results-{key}.json"))
        # results = load_json_file(os.path.join("/tmp/fragments", f"results-{key}.json"))
        #
        renderer = PdfRenderer(config={"preserve_interword_spaces": True})
        renderer.render(
            frames,
            results,
            output_filename=os.path.join(work_dir_icr, f"results-{key}.pdf"),
        )

        if False:
            renderer = TextRenderer(config={"preserve_interword_spaces": True})
            renderer.render(
                frames,
                results,
                output_filename=os.path.join(work_dir_icr, f"results-{key}.txt"),
            )
    except Exception as e:
        print("Error processing", img_path)
        print(e)
        raise e


if __name__ == "__main__":
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "~/tmp/4007/176073139.tif"

    # frames = [crop_to_content(frame, True) for frame in frames]

    use_cuda = torch.cuda.is_available()
    ocr_engine = DefaultOcrEngine(cuda=use_cuda)

    # check if we can process a single file or a directory
    if os.path.isdir(img_path):
        process_dir(ocr_engine, img_path)
    else:
        process_file(ocr_engine, img_path)
