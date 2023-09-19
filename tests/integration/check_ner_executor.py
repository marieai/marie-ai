import glob
import os
from typing import Dict

import transformers

from marie import Document, DocumentArray
from marie.constants import __config_dir__, __model_path__
from marie.conf.helper import load_yaml, storage_provider_config
from marie.executor.ner import NerExtractionExecutor
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage
from marie.logging.profile import TimeContext
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import docs_from_file
from marie.utils.image_utils import hash_bytes, hash_file
from marie.utils.json import store_json_object


def process_file(
        executor: NerExtractionExecutor,
        img_path: str,
        storage_enabled: bool,
        storage_conf: Dict[str, str],
):
    with TimeContext(f"### extraction info"):
        file = img_path.split("/")[-1]
        filename = file[-1].split(".")[0]
        checksum = hash_file(img_path)
        docs = docs_from_file(img_path)
        kwa = {"checksum": checksum, "img_path": img_path}
        payload = executor.extract(docs, **kwa)
        print(payload)
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
        executor: NerExtractionExecutor,
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
    _name_or_path = "/mnt/data/models/layoutlmv3-large-stride/checkpoint-1000"
    _name_or_path = "/mnt/data/marie-ai/model_zoo/rms/layoutlmv3-large-finetuned"
    _name_or_path = "/mnt/data/marie-ai/model_zoo/rms/layoutlmv3-large-patpay-ner"

    kwargs = {"__model_path__": __model_path__}
    _name_or_path = ModelRegistry.get(_name_or_path, **kwargs)

    # Load config
    config_data = load_yaml(os.path.join(__config_dir__, "marie-debug.yml"))
    storage_conf = storage_provider_config("postgresql", config_data)
    executor = NerExtractionExecutor(_name_or_path)

    storage_enabled = False
    img_path = f"~/datasets/private/medical_page_classification/raw/CORRESPONDENCE/174617756_2.tiff"
    img_path = f"~/datasets/private/medical_page_classification/raw/CHECK-FRONT-PATPAY/178870790_0.tiff"
    # img_path = f"/home/gbugaj/tmp/corr-indexing/small.png"

    if not os.path.isdir(img_path):
        process_file(executor, img_path, storage_enabled, storage_conf)
    else:
        process_dir(executor, img_path, storage_enabled, storage_conf)
