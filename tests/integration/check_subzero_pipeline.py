import os
from pprint import pprint
from typing import List

from marie.conf.helper import load_yaml
from marie.constants import __config_dir__
from marie.logging_core.mdc import MDC
from marie.logging_core.profile import TimeContext
from marie.pipe.extract_pipeline import split_filename
from marie.storage import StorageManager
from marie.storage.s3_storage import S3StorageHandler
from marie.subzero.engine.engine import SubzeroEngine
from marie.subzero.models.base import SelectorSet, TextSelector
from marie.subzero.models.definition import ExecutionContext, Layer, Template, WorkUnit
from marie.utils.docs import frames_from_file
from marie.utils.json import load_json_file


def setup_storage():
    """
    Setup storage manager
    """
    handler = S3StorageHandler(
        config={
            "S3_ACCESS_KEY_ID": "MARIEACCESSKEY",
            "S3_SECRET_ACCESS_KEY": "MARIESECRETACCESSKEY",
            "S3_STORAGE_BUCKET_NAME": "marie",
            "S3_ENDPOINT_URL": "http://localhost:8000",
            "S3_ADDRESSING_STYLE": "path",
        }
    )

    # export AWS_ACCESS_KEY_ID=MARIEACCESSKEY; export AWS_SECRET_ACCESS_KEY=MARIESECRETACCESSKEY;  aws s3 ls --endpoint-url http://localhost:8000
    StorageManager.register_handler(handler=handler)
    StorageManager.ensure_connection()


def load_from_annotation(file_path: str) -> tuple[dict, list]:
    """
    Load frames from annotation file
    :param file_path: file path
    :return: metadata, frames
    """
    meta_filename = os.path.splitext(file_path)[0] + ".json"

    frames = frames_from_file(file_path)
    metadata = load_json_file(meta_filename)

    return metadata, frames


if __name__ == "__main__":
    # setup_storage()
    # setup_torch_optimizations()

    MDC.put("request_id", "test")
    img_path = "~/datasets/private/corr-indexer/subz_meta/148445255_2.png"
    img_path = os.path.expanduser(img_path)

    # StorageManager.mkdir("s3://marie")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found : {img_path}")
    filename, prefix, suffix = split_filename(img_path)
    # s3_path = s3_asset_path(ref_id=filename, ref_type="pid", include_filename=True)
    # StorageManager.write(img_path, s3_path, overwrite=True)
    #
    config = load_yaml(
        os.path.join(
            __config_dir__, "tests-integration", "pipeline-classify-006.partial.yml"
        )
    )

    # runtime_conf = None

    def build_selector_sets(texts: List[str]) -> List[SelectorSet]:
        selectors = [
            TextSelector(tag=f"sel_{i}", text=text) for i, text in enumerate(texts)
        ]
        return [SelectorSet(selectors=selectors)]

    template = Template(tid="default_id", name="template_001", version=1)
    layer_1 = Layer()
    layer_1.start_selector_sets = build_selector_sets(["claim uid"])
    layer_1.stop_selector_sets = build_selector_sets(["Notes Claim UID"])

    template.add_layer(layer_1)

    # convert to yaml
    serialized = template.model_dump_json()
    pprint(serialized)

    if False:
        with TimeContext(f"### Subzero engine"):
            metadata, frames = load_from_annotation(img_path)
            work_unit = WorkUnit(
                doc_id=filename, template=template, frames=frames, metadata=metadata
            )
            contex = ExecutionContext.create(work_unit)
            print(contex)

            results = SubzeroEngine().match(contex)
            print(results)
            print("Completed")
