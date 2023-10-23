import glob
import glob
import os

from marie.executor.ner import NerExtractionExecutor
from marie.utils.docs import docs_from_file, frames_from_docs
from marie.utils.docs import frames_from_file
from marie.utils.image_utils import hash_frames_fast
from marie.utils.tiff_ops import merge_tiff_frames
from marie.utils.utils import ensure_exists

# executor = NerExtractionExecutor("rms/layoutlmv3-large-20221118-001-best")
# executor = NerExtractionExecutor("rms/layoutlmv3-large-corr")
executor = NerExtractionExecutor(
    "/home/gbugaj/dev/marieai/marie-ai/model_zoo/rms/layoutlmv3-large-20230711-stride128"
)


def process_image(img_path):
    # get name from filenama
    name = os.path.basename(img_path)
    name = os.path.splitext(name)[0]

    docs = docs_from_file(img_path)
    arr = frames_from_docs(docs)
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
    # process_image("/home/gbugaj/tmp/eob-issues/158954482_0.png")
