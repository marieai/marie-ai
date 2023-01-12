import os

import torch

from marie.boxes import PSMode
from marie.ocr import CoordinateFormat
from marie.ocr.extract_pipeline import ExtractPipeline
from marie.utils.docs import frames_from_file
from marie.utils.json import load_json_file
from marie.utils.utils import ensure_exists

if __name__ == "__main__":

    img_path = "/home/gbugaj/tmp/marie-cleaner/169150505/PID_1898_9172_0_169150505.tif"
    ocr_results = load_json_file(
        "/home/gbugaj/tmp/marie-cleaner/169150505/results.json"
    )

    print(ocr_results)
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    use_cuda = torch.cuda.is_available()
    frames = frames_from_file(img_path)

    filename = img_path.split("/")[-1]
    prefix = filename.split(".")[0]
    suffix = filename.split(".")[-1]

    def filename_supplier_page(
        filename: str, prefix: str, suffix: str, pagenumber: int
    ) -> str:
        return f"{prefix}_{pagenumber}.{suffix}"

    print(filename)
    print(prefix)
    print(suffix)

    for idx, frame in enumerate(frames):
        gen_name = filename_supplier_page(filename, prefix, suffix, idx + 1)
        print(gen_name)

    # pipeline = ExtractPipeline()
    # results = pipeline.execute(frames, PSMode.SPARSE, CoordinateFormat.XYWH)
