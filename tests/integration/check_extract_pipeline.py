import os

import torch

from marie.logging.profile import TimeContext
from marie.ocr.extract_pipeline import ExtractPipeline, split_filename, s3_asset_path
from marie.orchestrate.helper import generate_default_volume_and_workspace
from marie.storage import StorageManager
from marie.storage.s3_storage import S3StorageHandler
from marie.utils.docs import frames_from_file
from marie.utils.tiff_ops import merge_tiff_frames
from marie.utils.utils import ensure_exists


def setup_storage():
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
    setup_storage()

    img_path = (
        "~/datasets/dataset/medprov/PID/150300411/burst/PID_576_7188_0_150300411.tif"
    )

    img_path = "~/datasets/dataset/medprov/PID/150300431/PID_576_7188_0_150300431.tif"
    # img_path = "~/datasets/dataset/medprov/PID/169836035/PID_1007_7803_0_169836035.tif"
    # img_path = "~/datasets/dataset/medprov/PID/168483375/image8031655297039171311.pdf.merged.tif"
    # img_path = "~/datasets/dataset/medprov/PID/171131488/PID_1971_9380_0_171131488.tif"
    # img_path = "~/datasets/dataset/medprov/PID/168698807/image7494523172838185732.pdf.merged.tif"
    # img_path = "~/datasets/dataset/medprov/PID/169750819/PID_179_8268_0_169750819.tif"
    # img_path = "~/datasets/dataset/medprov/PID/169549209/PID_1015_7811_0_169549209.tif"
    # img_path = "~/datasets/dataset/medprov/PID/169837125/PID_1015_7811_0_169837125.tif"

    # # Register VFS handlers
    # # PathManager.register_handler(VolumeHandler(volume_base_dir="/home/greg/dataset/medprov/"))
    # PathManager.register(VolumeHandler(volume_base_dir="/home/gbugaj/datasets/medprov/"))
    # src_file = "volume://PID/150300431/PID_576_7188_0_150300431.tif"
    # process_workflow(src_file)

    img_path = os.path.expanduser(img_path)
    StorageManager.mkdir("s3://marie")

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    # remap TMP directory to workspace tmp directory
    if False:
        (
            generated_volumes,
            workspace_in_container,
        ) = generate_default_volume_and_workspace()
        gen_tmp_dir = generated_volumes[0].split(":")[0]
        os.environ["TMPDIR"] = ensure_exists(os.path.join(gen_tmp_dir, "tmp"))
        os.environ["TMP"] = ensure_exists(os.path.join(gen_tmp_dir, "tmp"))

    frames = frames_from_file(img_path)
    filename, prefix, suffix = split_filename(img_path)

    s3_path = s3_asset_path(ref_id=filename, ref_type="pid", include_filename=True)
    StorageManager.write(img_path, s3_path, overwrite=True)

    # resave PDF as tiff
    # merge_tiff_frames(frames, f"{img_path}.merged.tif")

    if True:
        pipeline = ExtractPipeline()
        with TimeContext(f"### ExtractPipeline info"):
            results = pipeline.execute(ref_id=filename, ref_type="pid", frames=frames)
