import glob
import os
from typing import Dict

import transformers

from marie import Document, DocumentArray
from marie.constants import __config_dir__, __model_path__
from marie.conf.helper import load_yaml, storage_provider_config
from marie.executor.ner import ChunkedNerExtractionExecutor
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage
from marie.logging.profile import TimeContext
from marie.registry.model_registry import ModelRegistry
from marie.utils.image_utils import hash_bytes, hash_file
from marie.utils.json import store_json_object


def process_file(
    executor: ChunkedNerExtractionExecutor,
    img_path: str,
    storage_enabled: bool,
    storage_conf: Dict[str, str],
):

    with TimeContext(f"### extraction info"):
        filename = img_path.split("/")[-1].replace(".png", "")
        checksum = hash_file(img_path)
        docs = None
        kwa = {"checksum": checksum, "img_path": img_path}
        payload = executor.extract(docs, **kwa)
        # print(payload)
        store_json_object(payload, f"/tmp/tensors/json/{filename}.json")

        if storage_enabled:
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


def process_dir(
    executor: ChunkedNerExtractionExecutor,
    image_dir: str,
    storage_enabled: bool,
    conf: Dict[str, str],
):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
        try:
            process_file(executor, img_path, storage_enabled, conf)
        except Exception as e:
            print(e)
            # raise e


if __name__ == "__main__":
    # 4.18.0  -> 4.21.0.dev0 : We should pin it to this version
    print(transformers.__version__)
    # _name_or_path = "rms/layoutlmv3-large-corr-ner"
    _name_or_path = "rms/layoutlmv3-large-20221118-001-best"
    _name_or_path = "rms/layoutlmv3-large-20221129-dit"
    kwargs = {"__model_path__": __model_path__}
    _name_or_path = ModelRegistry.get_local_path(_name_or_path, **kwargs)

    # Load config
    config_data = load_yaml(os.path.join(__config_dir__, "marie-debug.yml"))
    storage_conf = storage_provider_config("postgresql", config_data)
    executor = ChunkedNerExtractionExecutor(_name_or_path)

    storage_enabled = False
    img_path = f"/home/gbugaj/tmp/2022-08-09/PID_698_7367_0_159277012.tif"

    if not os.path.isdir(img_path):
        process_file(executor, img_path, storage_enabled, storage_conf)
    else:
        process_dir(executor, img_path, storage_enabled, storage_conf)
