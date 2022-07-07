import json

from marie.utils.docs import docs_from_file, array_from_docs
from marie.utils.image_utils import hash_file
from marie.utils.utils import ensure_exists
from marie.executor import TextExtractionExecutor, NerExtractionExecutor

if __name__ == "__main__":
    img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_1337_8147_0_156665066.png"
    # img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation_multipage/merged.tif"
    # img_path = f"/home/gbugaj/tmp/PID_1515_8370_0_157159253.tif"
    # img_path = f"/home/gbugaj/tmp/PID_1925_9291_0_157186552.tif"
    img_path = f"/home/gbugaj/tmp/PID_1925_9289_0_157186264.tif"

    checksum = hash_file(img_path)
    docs = None  # docs_from_file(img_path)
    kwa = {"checksum": checksum, "img_path": img_path}

    print(kwa)
    executor = NerExtractionExecutor()
    results = executor.extract(docs, **kwa)

    print(results)

