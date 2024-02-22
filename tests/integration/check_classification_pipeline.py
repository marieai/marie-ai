import os

from marie.conf.helper import load_yaml
from marie.constants import __config_dir__
from marie.logging.mdc import MDC
from marie.logging.profile import TimeContext
from marie.models.utils import setup_torch_optimizations
from marie.pipe.classification_pipeline import ClassificationPipeline
from marie.pipe.extract_pipeline import split_filename
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


if __name__ == "__main__":
    # setup_storage()
    setup_torch_optimizations()

    MDC.put("request_id", "test")
    img_path = "~/tmp/PID_1925_9289_0_157186264.tif"
    img_path = "~/tmp/188958112.tif"
    img_path = (
        # "~/datasets/private/corr-routing/Validation_Set/cms_letter/188172061_2.png"
        "~/datasets/private/corr-routing/faux/001/merged-001.tif"
        # "~/datasets/private/corr-routing/faux/002/merged-002.tif"
    )

    img_path = "~/tmp/PID_1925_9289_0_157186264.png"
    img_path = "~/datasets/private/corr-routing/jpmc_01-22-2024/ready/images/LEVEL_1/additional_information/08312023_734839_1_218_2.png"
    img_path = "~/datasets/private/corr-routing/jpmc_01-22-2024/ready/images/LEVEL_1/additional_information/09012023_734837_1_782___1_2.png"
    img_path = "~/datasets/private/corr-routing/jpmc_01-22-2024/ready/images/LEVEL_1/cms/08312023_772428_13_46_2.png"
    img_path = "~/datasets/private/corr-routing/jpmc_01-22-2024/ready/images/LEVEL_1/dispute/08312023_734913_1_4_2.png"
    img_path = "~/tmp/analysis/marie-issues/corr-failing/197333985/197333985.tif"
    img_path = "~/tmp/analysis/marie-issues/corr-failing/197107883/197107883.tif"
    img_path = "~/tmp/analysis/marie-issues/corr-failing/197333985/197333985-0001.png"
    img_path = "~/tmp/analysis/marie-issues/corr-failing/196675912/196675912.tif"

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

    pipelines_config = config["pipelines"]
    pipeline = ClassificationPipeline(pipelines_config=pipelines_config)

    runtime_conf = {
        # 'name': 'jpmc-corr'
    }

    # runtime_conf = None

    with TimeContext(f"### ClassificationPipeline info"):
        results = pipeline.execute(
            ref_id=filename, ref_type="pid", frames=frames_from_file(img_path), runtime_conf=runtime_conf
        )
        print(results)
