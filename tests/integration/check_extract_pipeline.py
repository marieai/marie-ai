import os

import torch

from marie.conf.helper import load_yaml
from marie.constants import __config_dir__, __model_path__
from marie.logging_core.mdc import MDC
from marie.logging_core.profile import TimeContext
from marie.models.utils import setup_torch_optimizations
from marie.pipe.extract_pipeline import ExtractPipeline, s3_asset_path, split_filename
from marie.storage import StorageManager
from marie.storage.s3_storage import S3StorageHandler
from marie.utils.docs import frames_from_file

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if torch.cuda.is_available() and torch.cuda.device_count() == 0:
    raise RuntimeError("CUDA is available but no GPU device is found, exiting...")


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

    img_path = "~/dev/rms/grapnel-g5/assets/TID-100985/226749569/PID_7350_14627_0_226749569.tif"

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

    pipeline_config = load_yaml(
        os.path.join(
            __config_dir__, "tests-integration", "pipeline-integration.partial.yml"
        )
    )

    use_cuda = torch.cuda.is_available()
    if use_cuda and torch.cuda.device_count() > 0:
        cuda_device = torch.cuda.current_device()
        print(f"Using CUDA device {cuda_device}")
        pipeline = ExtractPipeline(
            pipeline_config=pipeline_config["pipeline"], cuda=True
        )
    else:
        print("CUDA not available, using CPU")
        pipeline = ExtractPipeline(
            pipeline_config=pipeline_config["pipeline"], cuda=False
        )

    for i in range(1):
        with TimeContext(f"### ExtractPipeline info [{i}]"):
            results = pipeline.execute(
                ref_id=filename, ref_type="pid", frames=frames_from_file(img_path)
            )
            print(results)


def regions():
    img_path = "~/tmp/address-001.png"
    img_path = os.path.expanduser(img_path)
    # StorageManager.mkdir("s3://marie")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found : {img_path}")

    filename, prefix, suffix = split_filename(img_path)

    pipeline_config = load_yaml(
        os.path.join(
            __config_dir__, "tests-integration", "pipeline-integration.partial.yml"
        )
    )
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
            "h": 180,
        }
    ]
    regions = None

    for i in range(1):
        with TimeContext(f"### ExtractPipeline info [{i}]"):
            results = pipeline.execute(
                ref_id=filename,
                ref_type="pid",
                frames=frames_from_file(img_path),
                regions=regions,
            )
            print(results)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # os.environ["MARIE_DISABLE_CUDA"] = "True"
    torch.set_float32_matmul_precision("high")
    setup_torch_optimizations()

    run_extract_pipeline()
