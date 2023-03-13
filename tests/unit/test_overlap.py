import requests

import marie.helper
from marie import Flow
from marie.utils.overlap import merge_boxes
import numpy as np


def generate_bboxes(count: int):
    # generate random bounding boxes
    bboxes = []
    for i in range(count):
        x1 = np.random.randint(0, 1000)
        y1 = np.random.randint(0, 1000)
        bboxes.append(
            [x1, y1, x1 + np.random.randint(0, 1000), y1 + np.random.randint(0, 1000)]
        )
    return bboxes


def test_overlap_001():
    bboxes = generate_bboxes(10000)

    print(f"bboxes B : =============> {len(bboxes)}")
    # merge boxes with iou > 0.1 as they are likely to be the same box
    bboxes = merge_boxes(bboxes, 0.08)
    print(f"bboxes A : =============> {len(bboxes)}")

    assert len(bboxes) == 1
