import os
from pprint import pprint

from marie.conf.helper import load_yaml
from marie.constants import __config_dir__
from marie.logging_core.mdc import MDC
from marie.logging_core.profile import TimeContext
from marie.models.utils import setup_torch_optimizations
from marie.pipe.classification_pipeline import ClassificationPipeline
from marie.pipe.extract_pipeline import split_filename
from marie.storage import StorageManager
from marie.storage.s3_storage import S3StorageHandler
from marie.subzero.readers.meta_reader.meta_reader import MetaReader
from marie.utils.docs import frames_from_file
from marie.utils.json import load_json_file, store_json_object


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


def load_from_annotation(file_path:str):
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

    with TimeContext(f"### Subzero engine"):
        metadata, frames = load_from_annotation(img_path)
        doc = MetaReader.from_data(frames=frames, ocr_meta=metadata)
        print("Completed")
