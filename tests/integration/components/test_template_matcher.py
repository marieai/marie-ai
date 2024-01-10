import multiprocessing
import os
from typing import Tuple

import cv2
import numpy as np
import psutil
import torch

from marie.components.template_matching.dim_template_matching import (
    DeepDimTemplateMatcher,
)
from marie.logging.profile import TimeContext
from marie.registry.model_registry import ModelRegistry
from marie.utils.docs import docs_from_file, frames_from_docs
from marie.utils.json import load_json_file
from marie.utils.resize_image import resize_image


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
    # 0.1061895 empty image
    # 0.1061895 empty image
    for i in range(1):
        frames = frames_from_docs(
            docs_from_file("./assets/template_matching/sample-003.png")
        )
        frames_t = frames_from_docs(
            docs_from_file("./assets/template_matching/template-001.png")
        )
        frames_t = frames_from_docs(
            docs_from_file("./assets/template_matching/template-002.png")
        )

        template_coords = [[10, 135, 90, 30]]  #
        template_coords = [[195, 90, 90, 30]]  # -001
        template_coords = [[127, 92, 234, 27]]  # -002 - CLAIM PROVIDER
        template_coords = [[127, 92, 87, 27]]  # -002 - CLAIM

        template_bboxes = []
        template_frames = []
        template_labels = []

        for c in template_coords:
            x, y, w, h = c
            template = frames_t[0][y : y + h, x : x + w, :]
            # center the template in same size as the input image
            template, coord = resize_image(
                template,
                desired_size=(frames_t[0].shape[0], frames_t[0].shape[1]),
                color=(255, 255, 255),
                keep_max_size=True,
            )
            print(coord)
            template_frames.append(template)
            template_bboxes.append(coord)
            template_labels.append("test")
            cv2.imwrite(f"/tmp/dim/template.png", template)

        with TimeContext(f"Eval # {i}"):
            results = matcher.run(
                frames=frames,
                template_frames=template_frames,
                template_boxes=template_bboxes,
                template_labels=template_labels,
            )
            print(results)
