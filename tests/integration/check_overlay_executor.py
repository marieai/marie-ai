import glob
import os
from typing import Dict

from marie import Document, DocumentArray
from marie.conf.helper import load_yaml, storage_provider_config
from marie.constants import __config_dir__, __model_path__
from marie.executor.overlay.overlay_executor import OverlayExecutor
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage
from marie.logging_core.profile import TimeContext
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import docs_from_file
from marie.utils.image_utils import hash_file
from marie.utils.json import store_json_object
from marie.utils.utils import ensure_exists


def process_file(
    executor: OverlayExecutor,
    img_path: str,
    storage_enabled: bool,
    storage_conf: Dict[str, str],
):

    with TimeContext(f"### overlay info"):

        filename = img_path.split("/")[-1].replace(".png", "")
        print(f"Start processing file : {filename}")
        checksum = hash_file(img_path)
        docs = docs_from_file(img_path)
        parameters = {
            "ref_id": filename,
            "ref_type": "filename",
            "checksum": checksum,
            "img_path": img_path,
        }

        results = executor.segment(docs, parameters)
        print(results)
        ensure_exists("/tmp/tensors/json/")
        store_json_object(results, f"/tmp/tensors/json/{filename}.json")

        return results


def process_dir(
    executor: OverlayExecutor,
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
    _name_or_path = "default/overlay_test"
    # kwargs = {"__model_path__": __model_path__}

    # Load config
    config_data = load_yaml(os.path.join(__config_dir__, "marie-debug.yml"))
    storage_conf = storage_provider_config("postgresql", config_data)
    executor = OverlayExecutor(_name_or_path, True, storage_conf)

    storage_enabled = False
    img_path = f"277012.tif"

    if not os.path.isdir(img_path):
        process_file(executor, img_path, storage_enabled, storage_conf)
    else:
        process_dir(executor, img_path, storage_enabled, storage_conf)
