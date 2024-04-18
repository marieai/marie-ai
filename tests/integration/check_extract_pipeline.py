import os

import torch

from marie.conf.helper import load_yaml
from marie.constants import __config_dir__, __model_path__
from marie.logging.mdc import MDC
from marie.logging.profile import TimeContext
from marie.pipe.extract_pipeline import ExtractPipeline, s3_asset_path, split_filename
from marie.storage import StorageManager
from marie.storage.s3_storage import S3StorageHandler
from marie.utils.docs import frames_from_file


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


def run_extract_pipeline():
    # setup_storage()
    MDC.put("request_id", "test")
    img_path = "~/tmp/address-001.png"
    img_path = "~/tmp/demo/159000487_1.png"
    img_path = "~/tmp/4007/176075018.tif"
    img_path = os.path.expanduser(img_path)
    # StorageManager.mkdir("s3://marie")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found : {img_path}")

    filename, prefix, suffix = split_filename(img_path)
    print("Filename: ", filename)
    print("Prefix: ", prefix)
    print("Suffix: ", suffix)

    # s3_path = s3_asset_path(ref_id=filename, ref_type="pid", include_filename=True)
    # StorageManager.write(img_path, s3_path, overwrite=True)

    pipeline_config = load_yaml(os.path.join(__config_dir__, "tests-integration", "pipeline-integration.partial.yml"))
    pipeline = ExtractPipeline(pipeline_config=pipeline_config["pipeline"], cuda=True)

    with TimeContext(f"### ExtractPipeline info"):
        results = pipeline.execute(ref_id=filename, ref_type="pid", frames=frames_from_file(img_path))
        print(results)


def regions():
    img_path = "~/tmp/address-001.png"
    img_path = os.path.expanduser(img_path)
    # StorageManager.mkdir("s3://marie")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found : {img_path}")

    filename, prefix, suffix = split_filename(img_path)

    pipeline_config = load_yaml(os.path.join(__config_dir__, "tests-integration", "pipeline-integration.partial.yml"))
    # pipeline_config = load_yaml(os.path.join(__config_dir__, "tests-integration", "pipeline-integration-region.partial.yml"))
    pipeline = ExtractPipeline(pipeline_config=pipeline_config["pipeline"], cuda=True)
    regions = [
        {
            "mode": "sparse",
            "id": 12345,
            "pageIndex": 0,
            "x": 896,
            "y": 1562,
            "w": 733,
            "h": 180
        }
    ]

    for i in range(5):
        with TimeContext(f"### ExtractPipeline info [{i}]"):
            results = pipeline.execute(ref_id=filename, ref_type="pid", frames=frames_from_file(img_path),
                                       regions=regions)
            print(results)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # os.environ["MARIE_DISABLE_CUDA"] = "True"
    torch.set_float32_matmul_precision('high')

    run_extract_pipeline()
