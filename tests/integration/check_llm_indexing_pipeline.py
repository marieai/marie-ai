import os
from pprint import pprint

from marie.conf.helper import load_yaml
from marie.constants import __config_dir__
from marie.logging_core.mdc import MDC
from marie.logging_core.profile import TimeContext
from marie.models.utils import setup_torch_optimizations
from marie.pipe.llm_pipeline import LLMPipeline
from marie.utils.docs import frames_from_file
from marie.utils.json import store_json_object

if __name__ == "__main__":

    setup_torch_optimizations()
    MDC.put("request_id", "test")

    # Pipeline setup
    config = load_yaml(
        os.path.join(
            __config_dir__, "tests-integration", "pipeline-indexing.partial.yml"
        )
    )
    pprint(config)
    pipelines_config = config.get("pipelines")
    pipeline = LLMPipeline(pipelines_config=pipelines_config)

    runtime_conf = {
        # Different Runtime Configs here
    }

    dataset_path = os.path.expanduser("~/data/corr/corr_extract/images")
    output_path = os.path.expanduser("~/data/corr/corr_extract/results")
    images = os.listdir(dataset_path)
    with TimeContext(f"### LLM Indexing Pipeline"):
        for filename in images:
            # Input setup
            img_path = os.path.join(dataset_path, filename)
            results = pipeline.execute(
                ref_id=filename,
                ref_type="pid",
                frames=frames_from_file(img_path),
                runtime_conf=runtime_conf,
            )
            print(results)
            store_json_object(results, os.path.join(output_path, f"{filename}.meta.json"))


