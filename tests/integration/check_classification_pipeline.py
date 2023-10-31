import os

from transformers import LongformerTokenizer, AutoTokenizer

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
            "S3_ENDPOINT_URLXX": "http://localhost:8000",
            "S3_ENDPOINT_URL": "http://gext-05.rms-asp.com:8000",
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
    img_path = os.path.expanduser(img_path)
    # StorageManager.mkdir("s3://marie")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found : {img_path}")

    filename, prefix, suffix = split_filename(img_path)

    # s3_path = s3_asset_path(ref_id=filename, ref_type="pid", include_filename=True)
    # StorageManager.write(img_path, s3_path, overwrite=True)

    pipeline_config = load_yaml(
        os.path.join(
            __config_dir__, "tests-integration", "pipeline-classify-001.partial.yml"
        )
    )
    pipeline = ClassificationPipeline(pipeline_config=pipeline_config["pipeline"])

    with TimeContext(f"### ClassificationPipeline info"):
        results = pipeline.execute(
            ref_id=filename, ref_type="pid", frames=frames_from_file(img_path)
        )
