import json
import os

from marie.executor.text import TextExtractionExecutor
from marie.executor.ner.utils import visualize_icr
from marie.numpyencoder import NumpyEncoder
from marie.renderer import PdfRenderer, TextRenderer
from marie.utils.docs import array_from_docs, docs_from_file
from marie.utils.json import load_json_file, store_json_object
from marie.utils.utils import ensure_exists
from marie.constants import __model_path__

if __name__ == "__main__":

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "./assets/psm/word/0001.png"
    # img_path = "./assets/english/Scanned_documents/Picture_029.tif"
    # img_path = "./assets/english/Scanned_documents/Picture_010.tif"
    img_path = "./assets/english/Lines/002.png"
    img_path = "/home/gbugaj/dataset/funsd/dataset/training_data/images/00040534.png"
    img_path = "/home/gbugaj/clean_medical/PID_1038_7836_0_149512505_page_0021.tif"

    docs = docs_from_file(img_path)
    frames = array_from_docs(docs)
    kwa = {"payload": {"output": "json", "mode": "line", "format": "xyxy"}}
    kwa = {"payload": {"output": "json", "mode": "sparse", "format": "xywh"}}
    # kwa = {"payload": {"output": "json", "mode": "line"}}

    if True:
        executor = TextExtractionExecutor()
        results = executor.extract(docs, parameters=kwa)

        print(results)
        store_json_object(results, os.path.join("/tmp/fragments", "results.json"))

    results = load_json_file(os.path.join("/tmp/fragments", "results.json"))
    visualize_icr(frames, results)

    renderer = TextRenderer(config={"preserve_interword_spaces": True})
    renderer.render(
        frames, results, output_filename=os.path.join(work_dir_icr, "results.txt")
    )

    if True:
        renderer = PdfRenderer(config={})
        renderer.render(
            frames, results, output_filename=os.path.join(work_dir_icr, "results.pdf")
        )