import multiprocessing
import os
import random
from typing import Tuple

import cv2
import numpy as np
import psutil
import torch

from marie.components.template_matching.composite_template_maching import (
    CompositeTemplateMatcher,
)
from marie.components.template_matching.meta_template_matching import (
    MetaTemplateMatcher,
)
from marie.components.template_matching.vqnnf_template_matching import (
    VQNNFTemplateMatcher,
)
from marie.logging.profile import TimeContext
from marie.utils.docs import docs_from_file, frames_from_docs
from marie.utils.json import load_json_file
from marie.utils.resize_image import resize_image


# extract windows snippet from the input image centered around the template
def extract_windows(
    image: np.ndarray,  # h, w, c
    template_frames: Tuple[np.ndarray],  # h, w, c
    template_bboxes: Tuple[Tuple[int]],  # x, y, w, h
    window_size: Tuple[int],  # h, w
):
    windows = []
    for template_frame, template_bbox in zip(template_frames, template_bboxes):
        x, y, w, h = template_bbox
        # extract window from the input image
        window = image[y : y + h, x : x + w, :]
        # resize the window to the desired size
        window, _ = resize_image(
            window,
            desired_size=window_size,
            color=(255, 255, 255),
            keep_max_size=True,
        )
        # center the template in same size as the input image
        window, _ = resize_image(
            window,
            desired_size=(image.shape[0], image.shape[1]),
            color=(255, 255, 255),
            keep_max_size=True,
        )
        windows.append(window)
    return windows


def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def test_template_matcher():
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
            # docs_from_file("./assets/template_matching/sample-001-95_percent.png")
            docs_from_file("./assets/template_matching/sample-002.png")
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
        }

        key = "002-A"
        key = "002-B"
        # key = "002-A"

        template_coords = samples[key]["coords"]
        frames_t = frames_from_docs(docs_from_file(samples[key]["image"]))

        template_bboxes = []
        template_frames = []
        template_labels = []
        template_texts = []

        if False:
            # test window size
            template_coords = [[215, 175, 124, 35]]
            frames = frames_from_docs(
                docs_from_file("./assets/template_matching/sample-003.png")
            )

            window_size = (500, 300)  # h, w

            print("frames", frames[0].shape)
            for idx, c in enumerate(template_coords):
                frame = frames[idx]
                x, y, w, h = c
                template = frames[0][y : y + h, x : x + w, :]

                cx = x + w // 2
                cy = y + h // 2

                wh = window_size[0] // 2
                ww = window_size[1] // 2

                # clip the template from the input image expanded by the window size centered around the template
                frame_shape = frames[0].shape
                x1 = max(0, cx - ww // 2)
                y1 = max(0, cy - wh // 2)
                w1 = min(frame_shape[1], ww)
                h1 = min(frame_shape[0], wh)
                window = frame[y1 : y1 + h1, x1 : x1 + w1, :]

                cv2.imwrite(f"/tmp/dim/template.png", template)
                cv2.imwrite(f"/tmp/dim/window.png", window)

                # if the window is smaller than the desired size, resize it
                if window.shape[0] < window_size[0] or window.shape[1] < window_size[1]:
                    window, _ = resize_image(
                        window,
                        desired_size=window_size,
                        color=(255, 255, 255),
                        keep_max_size=True,
                    )
                cv2.imwrite(f"/tmp/dim/window_resized.png", window)

        window_size = (128, 384)
        # window_size = (384, 384)
        window_size = (512, 512)

        for c in template_coords:
            x, y, w, h = c
            template = frames_t[0]
            coord = c

            if False:
                template = frames_t[0][y : y + h, x : x + w, :]
                # center the template in same size as the input image
                template, coord = resize_image(
                    template,
                    desired_size=window_size,  # (frames_t[0].shape[0], frames_t[0].shape[1]),
                    color=(255, 255, 255),
                    keep_max_size=True,
                )

        for c in template_coords:
            x, y, w, h = c
            template = frames_t[0][y : y + h, x : x + w, :]
            # center the template in same size as the input image
            template, coord = resize_image(
                template,
                desired_size=window_size,  # (frames_t[0].shape[0], frames_t[0].shape[1]),
                color=(255, 255, 255),
                keep_max_size=True,
            )

            template_frames.append(template)
            template_bboxes.append(coord)
            template_labels.append("test")
            template_texts.append(samples[key].get("text", ""))

            cv2.imwrite(f"/tmp/dim/template.png", template)

        with TimeContext(f"Eval # {i}"):
            results = matcher.run(
                frames=frames,
                # TODO: convert to Pydantic model
                template_frames=template_frames,
                template_boxes=template_bboxes,
                template_labels=template_labels,
                template_texts=template_texts,
                metadata=ocr_results,
                score_threshold=0.80,
                max_overlap=0.5,
                max_objects=2,
                window_size=window_size,
                downscale_factor=1,
            )
            print(results)
