import os

from marie.conf.helper import load_yaml
from marie.logging.mdc import MDC
from marie.logging.profile import TimeContext
from marie.ocr.extract_pipeline import ExtractPipeline, split_filename, s3_asset_path
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
    MDC.put("request_id", "test")
    img_path = "~/tmp/txt_overlay001.jpg"
    img_path = "~/tmp/address-001.png"
    img_path = os.path.expanduser(img_path)
    # StorageManager.mkdir("s3://marie")

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    frames = frames_from_file(img_path)
    # frames = [frames[1]]
    filename, prefix, suffix = split_filename(img_path)

    from marie.constants import __model_path__, __config_dir__

    pipeline_config = load_yaml(os.path.join(__config_dir__, "tests-integration", "pipeline-integration.partial.yml"))
    pipeline_config = pipeline_config["pipeline"]
    pipeline = ExtractPipeline(pipeline_config=pipeline_config)

    # s3_path = s3_asset_path(ref_id=filename, ref_type="pid", include_filename=True)
    # StorageManager.write(img_path, s3_path, overwrite=True)
    # resave PDF as tiff
    # merge_tiff_frames(frames, f"{img_path}.merged.tif")

    if True:
        with TimeContext(f"### ExtractPipeline info"):
            results = pipeline.execute(ref_id=filename, ref_type="pid", frames=frames)
