import os

from marie.logging.mdc import MDC
from marie.logging.profile import TimeContext
from marie.ocr.extract_pipeline import ExtractPipeline, split_filename, s3_asset_path
from marie.storage import StorageManager
from marie.storage.s3_storage import S3StorageHandler
from marie.utils.docs import frames_from_file


def setup_storage():
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
    MDC.put('request_id', 'test')

    img_path = (
        # "~/dev/ldt-document-dump/cache/179575453.tif"
        # "~/tmp/eob-issues/158954482_0.png"
        # "~/tmp/analysis/PID_1956_9362_0_177978797/300DPI/PID_1956_9362_0_177978797.tif"
        # "~/tmp/analysis/DEVOPSSD-54421/178443716.tif"
        # "~/dev/ldt-document-dump/cache/181313152.tif"
        "~/tmp/failures-marie/snippets/overlay_image_1_10666343744.png"
    )

    # # Register VFS handlers
    # # PathManager.register_handler(VolumeHandler(volume_base_dir="/home/greg/dataset/medprov/"))
    # PathManager.register(VolumeHandler(volume_base_dir="/home/gbugaj/datasets/medprov/"))
    # src_file = "volume://PID/150300431/PID_576_7188_0_150300431.tif"
    # process_workflow(src_file)

    img_path = os.path.expanduser(img_path)
    # StorageManager.mkdir("s3://marie")

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    frames = frames_from_file(img_path)
    # frames = [frames[1]]
    filename, prefix, suffix = split_filename(img_path)
    pipeline = ExtractPipeline()

    # s3_path = s3_asset_path(ref_id=filename, ref_type="pid", include_filename=True)
    # StorageManager.write(img_path, s3_path, overwrite=True)
    # resave PDF as tiff
    # merge_tiff_frames(frames, f"{img_path}.merged.tif")

    if True:
        with TimeContext(f"### ExtractPipeline info"):
            results = pipeline.execute(ref_id=filename, ref_type="pid", frames=frames)
