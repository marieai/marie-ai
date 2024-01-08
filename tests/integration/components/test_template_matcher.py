import multiprocessing
import os

import cv2
import psutil
import torch

from marie.components.template_matching.dim_template_matching import (
    DeepDimTemplateMatcher,
)
from marie.logging.profile import TimeContext
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import docs_from_file, frames_from_docs
from marie.utils.json import load_json_file


def test_template_matcher():
    # kwargs = {"__model_path__": __model_path__}
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    import intel_extension_for_pytorch as ipex
    print("ipex", ipex.__version__)
    print("torch", torch.__version__)

    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OMP_SCHEDULE"] = "STATIC"
    os.environ["OMP_PROC_BIND"] = "CLOSE"
    os.environ["OMP_PLACES"] = "CORES"
    # set to core-count of your CPU
    torch.set_num_threads(psutil.cpu_count(logical=False))

    matcher = DeepDimTemplateMatcher(model_name_or_path="vgg19")

    for i in range(1):
        documents = docs_from_file("./assets/template_matching/sample-003.png")
        frames = frames_from_docs(documents)

        template_coords = [[45, 296, 231, 27]]  # -002
        template_coords = [[9, 43, 228, 30]]  # -002

        templates = []
        for c in template_coords:
            x, y, w, h = c
            template = frames[0][y: y + h, x: x + w, :]
            templates = [template]
            cv2.imwrite(f"/tmp/dim/template.png", template)

        with TimeContext(f"Eval # {i}"):
            results = matcher.run(frames=frames, templates=templates, labels=["test"] * len(frames))
            print(results)
