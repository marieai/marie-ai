import glob
import os

from marie.executor import NerExtractionExecutor
from marie.utils.image_utils import hash_file
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
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.png"))):
        try:
            process_file(executor, img_path)
            break
        except Exception as e:
            print(e)
            # raise e


if __name__ == "__main__":
    ensure_exists("/tmp/tensors/json")

    executor = NerExtractionExecutor()
    # process_dir(executor, "/home/greg/dataset/assets-private/corr-indexer/validation/")

    if True:
        img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_718_7393_0_156664823.png"
        # img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation_multipage/merged.tif"
        # img_path = f"/home/gbugaj/tmp/PID_1515_8370_0_157159253.tif"
        # img_path = f"/home/gbugaj/tmp/PID_1925_9291_0_157186552.tif"
        img_path = f"/home/gbugaj/tmp/PID_1925_9289_0_157186264.tif"
        img_path = f"/home/greg/tmp/PID_1925_9289_0_157186264.png" # Invalid token marking
        img_path = f"/home/gbugaj/tmp/PID_1925_9289_0_157186264.tif" # Invalid token marking

        process_file(executor, img_path)
