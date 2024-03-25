import multiprocessing
import os
import random

import cv2
import numpy as np
import psutil
import torch
from PIL import Image

from marie.components.template_matching import (
    BaseTemplateMatcher,
    CompositeTemplateMatcher,
    MetaTemplateMatcher,
    VQNNFTemplateMatcher,
)
from marie.logging.profile import TimeContext
from marie.utils.docs import docs_from_file, frames_from_docs
from marie.utils.json import load_json_file
from marie.utils.resize_image import resize_image, resize_image_progressive


def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_template_matcher():
    # kwargs = {"__model_path__": __model_path__}
    setup_seed(42)

    import intel_extension_for_pytorch as ipex

    print("ipex", ipex.__version__)
    print("torch", torch.__version__)

    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OMP_SCHEDULE"] = "STATIC"
    os.environ["OMP_PROC_BIND"] = "CLOSE"
    os.environ["OMP_PLACES"] = "CORES"
    # set to core-count of your CPU
    torch.set_num_threads(psutil.cpu_count(logical=False))

    # matcher = DeepDimTemplateMatcher(model_name_or_path="vgg19")
    matcher_vqnnft = VQNNFTemplateMatcher(model_name_or_path="NONE")
    matcher_meta = MetaTemplateMatcher(model_name_or_path="NONE")
    matcher = CompositeTemplateMatcher(matchers=[matcher_vqnnft, matcher_meta])

    matcher = matcher_vqnnft
    # matcher = CompositeTemplateMatcher(matchers=[matcher_vqnnft])

    for i in range(1):
        ocr_results = load_json_file(
            "./assets/template_matching/sample-001.png.meta.json"
        )
        ocr_results = None
        frames = frames_from_docs(
            # docs_from_file("./assets/template_matching/sample-001-exact.png")
            # docs_from_file("./assets/template_matching/sample-005.png")
            # docs_from_file("./assets/template_matching/sample-001.png")
            docs_from_file("./assets/template_matching/sample-006.png")
            # docs_from_file("./assets/template_matching/sample-001-95_percent.png")
            # docs_from_file("./assets/template_matching/sample-002.png")
            # docs_from_file("/home/gbugaj/tmp/medrx/pid/173358514/PID_749_7449_0_157676683.png")
        )

        samples = {
            "001-A": {
                "label": "CLAIM #:",
                "coords": [[175, 90, 126, 30]],  # -001 CLAIM #:
                "image": "./assets/template_matching/template-001.png",
            },
            "001-B": {
                "label": "#:",
                "coords": [[265, 90, 35, 30]],  # -001 #:
                "image": "./assets/template_matching/template-001.png",
            },
            "001-C": {
                "label": ":",
                "coords": [[290, 90, 35, 30]],  # -001 :
                "image": "./assets/template_matching/template-001.png",
            },
            "002-A": {
                "label": "CLAIM PROVIDER",
                "coords": [[127, 92, 234, 27]],  # -002 :
                "image": "./assets/template_matching/template-002.png",
                "text": "CLAIM PROVIDER :",
            },
            "002-B": {
                "label": "CLAIM",
                "coords": [[127, 92, 87, 27]],  # -002 :
                "image": "./assets/template_matching/template-002.png",
            },
            "004-A": {
                "label": "block",
                "coords": [[182, 29, 145, 168]],  # -004 :
                "image": "./assets/template_matching/template-004.png",
            },
            "005-A": {
                "label": "CLAIM",
                "coords": [[218, 181, 89, 31]],  # -004 :
                "image": "./assets/template_matching/template-005.png",
                "target": "./assets/template_matching/sample-005.png",
            },
            "006-A": {
                "label": "NPI #:",
                "coords": [[1010, 134, 101, 33]],  # -004 :
                "image": "./assets/template_matching/sample-005.png",
                "target": "./assets/template_matching/sample-005.png",
            },
            "006-B": {
                "label": "Claim #:",
                "coords": [[223, 643, 124, 38]],  # -004 :
                "image": "./assets/template_matching/sample-005.png",
                "target": "./assets/template_matching/sample-005.png",
            },

            "007-B": {
                "label": "CLAIM",
                "coords": [[176, 90, 90, 33]],
                "image": "./assets/template_matching/template-005.png",
                "target": "./assets/template_matching/sample-006.png",
            },
        }

        key = "007-B"

        template_coords = samples[key]["coords"]
        frames_t = frames_from_docs(docs_from_file(samples[key]["image"]))

        template_bboxes = []
        template_frames = []
        template_labels = []
        template_texts = []

        # H, W
        window_size = (512, 512)
        window_size = [384, 384]

        template_frames, template_bboxes = BaseTemplateMatcher.extract_windows(
            frames_t[0], template_coords, window_size, allow_padding=True
        )

        for template_frame in template_frames:
            cv2.imwrite(f"/tmp/dim/template/template_frame_XXX_{i}.png", template_frame)

        for c in template_coords:
            x, y, w, h = c
            template = frames_t[0][y: y + h, x: x + w, :]
            # center the template in same size as the input image
            template, coord = resize_image(
                template,
                desired_size=window_size,  # (frames_t[0].shape[0], frames_t[0].shape[1]),
                color=(255, 255, 255),
                keep_max_size=True,
            )

            # template_frames.append(template)
            # template_bboxes.append(coord)
            template_labels.append("test")
            template_texts.append(samples[key].get("text", ""))
            # cv2.imwrite(f"/tmp/dim/template.png", template)
        for k in range(1):
            with TimeContext(f"Eval # {i}"):
                results = matcher.run(
                    frames=frames,
                    template_frames=template_frames,
                    template_boxes=template_bboxes,
                    template_labels=template_labels,
                    template_texts=template_texts,
                    metadata=ocr_results,
                    score_threshold=0.80,
                    max_overlap=0.5,
                    max_objects=2,
                    window_size=window_size,
                    downscale_factor=.75,
                )
                print(results)


if __name__ == "__main__":
    check_template_matcher()
