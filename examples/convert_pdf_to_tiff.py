import base64
import glob
import os
import time
import uuid

import requests
from PIL import Image

from marie.executor.ner.utils import visualize_icr
from marie.utils.docs import frames_from_file
from marie.utils.tiff_ops import merge_tiff_frames
from marie.utils.utils import ensure_exists


from marie.utils.json import load_json_file, store_json_object

from PIL import Image

from marie.executor.ner import NerExtractionExecutor
from marie.utils.docs import docs_from_file, array_from_docs
from marie.utils.image_utils import hash_file, hash_frames_fast

executor = NerExtractionExecutor("rms/layoutlmv3-large-20221118-001-best")
# executor = NerExtractionExecutor("rms/layoutlmv3-large-corr")


def process_image(img_path):
    # get name from filenama
    name = os.path.basename(img_path)
    name = os.path.splitext(name)[0]

    docs = docs_from_file(img_path)
    arr = array_from_docs(docs)
    checksum = hash_frames_fast(arr)
    kwa = {}
    results = executor.extract(docs, **kwa)
    print(results)
    # store_json_object(results, f"/tmp/pdf_2_tif/json/{name}.json")
    return results


def process_dir_ner(image_dir: str):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.tif"))):
        try:
            print(img_path)
            process_image(img_path)
        except Exception as e:
            print(e)
            # raise e


def process_dir_pdf(image_dir: str):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
        try:
            print(img_path)
            # get name from filenama
            name = os.path.basename(img_path)
            name = os.path.splitext(name)[0]
            print(name)
            # read fraes from pdf
            frames = frames_from_file(img_path)
            print(len(frames))
            merge_tiff_frames(
                frames, "/home/gbugaj/tmp/corr-routing/finished/V20/{}.tif".format(name)
            )
        except Exception as e:
            print(e)
            # raise e


if __name__ == "__main__":
    ensure_exists("/tmp/pdf_2_tif")

    # process_dir_pdf("/home/gbugaj/tmp/corr-routing/finished/V20_LARGE")
    # process_dir("/opt/shares/_hold/ENSEMBLE/SAMPLE/PRODUCTION/PDF")
    # process_dir_ner("/tmp/pdf_2_tif")
    # process_image("/home/gbugaj/tmp/analysis/OVERFLOWING-CORR/148447127_0.png")
    process_image("/home/gbugaj/tmp/PID_1925_9289_0_157186264.png")
