import json

from marie.executor import TextExtractionExecutor
from marie.numpyencoder import NumpyEncoder
from marie.utils.docs import docs_from_file, array_from_docs
from marie.utils.utils import ensure_exists

if __name__ == "__main__":

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "./assets/psm/word/0001.png"
    # img_path = "./assets/english/Scanned_documents/Picture_029.tif"
    # img_path = "./assets/english/Scanned_documents/Picture_010.tif"
    img_path = "./assets/english/Lines/002.png"

    docs = docs_from_file(img_path)
    frames = array_from_docs(docs)
    kwa = {"payload": {"output": "json", "mode": "line", "format": "xyxy"}}
    # kwa = {"payload": {"output": "json", "mode": "line"}}
    executor = TextExtractionExecutor()
    results = executor.extract(docs, **kwa)

    print(results)

    with open("/home/greg/.marie/xyxy.json", "w") as json_file:
        json.dump(
            results,
            json_file,
            sort_keys=False,
            separators=(",", ": "),
            ensure_ascii=False,
            indent=4,
            cls=NumpyEncoder,
        )
