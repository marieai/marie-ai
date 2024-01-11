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


# extract windows snippet from the input image centered around the template
def extract_windows(
        image: np.ndarray,
        template_frames: Tuple[np.ndarray],
        template_bboxes: Tuple[Tuple[int]],
        window_size: Tuple[int],
):
    windows = []
    for template_frame, template_bbox in zip(template_frames, template_bboxes):
        x, y, w, h = template_bbox
        # extract window from the input image
        window = image[y: y + h, x: x + w, :]
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
        frames_t = frames_from_docs(
            docs_from_file("./assets/template_matching/template-004.png")
        )

        template_coords = [[10, 135, 90, 30]]  #
        # template_coords = [[175, 90, 126, 30]]  # -001 CLAIM #:
        # template_coords = [[265, 90, 35, 30]]  # -001  #:
        # template_coords = [[290, 90, 35, 30]]  # -001  :
        template_coords = [[127, 92, 234, 27]]  # -002 - CLAIM PROVIDER
        # template_coords = [[127, 92, 87, 27]]  # -002 - CLAIM
        template_coords = [[182, 29, 145, 168]]  # -004 - block

        template_bboxes = []
        template_frames = []
        template_labels = []

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
                template = frames[0][y: y + h, x: x + w, :]

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
                window = frame[y1: y1 + h1, x1: x1 + w1, :]

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


        for c in template_coords:
            x, y, w, h = c
            template = frames_t[0][y: y + h, x: x + w, :]
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
