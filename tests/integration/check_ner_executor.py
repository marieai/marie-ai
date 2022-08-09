import glob
import os
from typing import Dict

import transformers

from marie.conf.helper import storage_provider_config, load_yaml
from marie.executor import NerExtractionExecutor
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage
from marie.logging.profile import TimeContext
from marie.registry.model_registry import ModelRegistry
from marie.utils.image_utils import hash_file, hash_bytes
from marie.utils.json import store_json_object
from marie import (
    Document,
    DocumentArray,
    __model_path__,
    __config_dir__,
)


def process_file(
    executor: NerExtractionExecutor, img_path: str, storage_conf: Dict[str, str]
):

    with TimeContext(f"### extraction info"):
        filename = img_path.split("/")[-1].replace(".png", "")
        checksum = hash_file(img_path)
        docs = None
        kwa = {"checksum": checksum, "img_path": img_path}
        payload = executor.extract(docs, **kwa)
        print(payload)
        store_json_object(payload, f"/tmp/tensors/json/{filename}.json")

        if True:
            storage = PostgreSQLStorage(
                hostname=storage_conf["hostname"],
                port=int(storage_conf["port"]),
                username=storage_conf["username"],
                password=storage_conf["password"],
                database=storage_conf["database"],
                table="check_ner_executor",
            )

            dd2 = DocumentArray([Document(content=payload)])

            storage.add(dd2, {"ref_id": filename, "ref_type": "filename"})

        return payload


def process_dir(executor: NerExtractionExecutor, conf: Dict[str, str], image_dir: str):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
        try:
            process_file(executor, img_path, conf)
        except Exception as e:
            print(e)
            # raise e


if __name__ == "__main__":

    # pip install git+https://github.com/huggingface/transformers
    # 4.18.0  -> 4.21.0.dev0 : We should pin it to this version
    print(transformers.__version__)
    _name_or_path = "rms/layoutlmv3-large-corr-ner"
    kwargs = {"__model_path__": __model_path__}
    _name_or_path = ModelRegistry.get_local_path(_name_or_path, **kwargs)

    print(__config_dir__)
    # Load config
    config_data = load_yaml(os.path.join(__config_dir__, "marie-debug.yml"))
    storage_conf = storage_provider_config("postgresql", config_data)

    executor = NerExtractionExecutor(_name_or_path)

    # process_dir(executor, "/home/greg/dataset/assets-private/corr-indexer/validation/")
    # process_dir(executor, storage_conf, "/home/gbugaj/tmp/medrx-missing-corr/")
    process_dir(executor, storage_conf, "/home/gbugaj/tmp/2022-08-09")

    if True:
        img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_162_6505_0_156695212.png"
        # img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_1898_9200_0_156692336.png"
        img_path = f"/home/gbugaj/tmp/medrx-missing-corr/PID_1055_7854_0_158147069.tif"

        # process_file(executor, img_path, storage_conf)
