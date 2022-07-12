import glob
import os
import time

import numpy as np

from marie.executor import NerExtractionExecutor
from marie.utils.docs import load_image, docs_from_file, array_from_docs
from marie.utils.image_utils import hash_file, hash_bytes
from marie.utils.json import store_json_object
from marie.utils.utils import ensure_exists


def process_file(executor: NerExtractionExecutor, img_path: str):
    filename = img_path.split("/")[-1].replace(".png", "")
    checksum = hash_file(img_path)
    docs = None
    kwa = {"checksum": checksum, "img_path": img_path}
    results = executor.extract(docs, **kwa)

    print(results)
    store_json_object(results, f"/tmp/tensors/json/{filename}.json")
    return results


def process_dir(executor: NerExtractionExecutor, image_dir: str):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.tif"))):
        try:
            process_file(executor, img_path)
        except Exception as e:
            print(e)
            # raise e


if __name__ == "__main__":
    ensure_exists("/tmp/tensors/json")
    executor = NerExtractionExecutor()
    # process_dir(executor, "/home/greg/dataset/assets-private/corr-indexer/validation/")
    # process_dir(executor, "/home/gbugaj/tmp/medrx")

    if True:
        img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_718_7393_0_156664823.png"
        # img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation_multipage/merged.tif"
        # img_path = f"/home/gbugaj/tmp/PID_1515_8370_0_157159253.tif"
        img_path = f"/home/gbugaj/tmp/PID_1925_9291_0_157186552.tif"
        img_path = f"/home/gbugaj/tmp/PID_1925_9289_0_157186264.tif"
        img_path = f"/home/gbugaj/tmp/medrx/PID_1313_8120_0_157638578.tif"
        # img_path = f"/home/gbugaj/tmp/PID_1925_9289_0_157186264.tif"
        # img_path = (
        #     f"/home/greg/tmp/PID_1925_9289_0_157186264.png"  # Invalid token marking
        # )
        # img_path = (
        #     f"/home/gbugaj/tmp/PID_1925_9289_0_157186264.tif"  # Invalid token marking
        # )
        # img_path = (
        #     f"/home/greg/tmp/PID_1925_9289_0_157186264.tif"  # Invalid token marking
        # )
        # img_path = f"/home/greg/tmp/image8918637216567684920.pdf"
        # img_path = f"/home/greg/tmp/PID_1925_9289_0_157186264.png"

        docs = docs_from_file(img_path)
        frames = array_from_docs(docs)

        time_nanosec = time.time_ns()
        src = []
        for i, frame in enumerate(frames):
            src = np.append(src, np.ravel(frame))
        checksum = hash_bytes(src)
        print(checksum)
        time_nanosec = (time.time_ns() - time_nanosec) / 1000000000
        print(time_nanosec)

        process_file(executor, img_path)
