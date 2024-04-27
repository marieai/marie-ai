import os

import torch

from marie.components.template_matching.document_matched import (
    load_template_matching_definitions,
    match_templates,
)
from marie.conf.helper import load_yaml
from marie.constants import __config_dir__, __model_path__
from marie.logging.mdc import MDC
from marie.logging.profile import TimeContext
from marie.pipe.extract_pipeline import ExtractPipeline, s3_asset_path, split_filename
from marie.storage import StorageManager
from marie.storage.s3_storage import S3StorageHandler
from marie.utils.docs import frames_from_file


def run_extract_pipeline():
    MDC.put("request_id", "test")
    img_path = "~/tmp/analysis/document-boundary/samples/PID_808_7548_0_202343052.tif"
    img_path = os.path.expanduser(img_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found : {img_path}")

    frames = frames_from_file(img_path)
    filename, prefix, suffix = split_filename(img_path)
    print("Filename: ", filename)
    print("Prefix: ", prefix)
    print("Suffix: ", suffix)

    pipeline_config = load_yaml(os.path.join(__config_dir__, "tests-integration", "pipeline-integration.partial.yml"))
    pipeline = ExtractPipeline(pipeline_config=pipeline_config["pipeline"], cuda=True)

    runtime_conf = {
        "template_matching": {
            "definition_id": "120791",
        }
    }

    with TimeContext(f"### ExtractPipeline info"):
        # for k in range(0, 3):
        results = pipeline.execute(ref_id=filename, ref_type="pid", frames=frames_from_file(img_path),
                                   runtime_conf=runtime_conf)
        print(results)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # os.environ["MARIE_DISABLE_CUDA"] = "True"
    torch.set_float32_matmul_precision('high')

    run_extract_pipeline()
