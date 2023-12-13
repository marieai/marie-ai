import timeit

import numpy as np

from marie.utils.overlap import merge_boxes


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
    bboxes = generate_bboxes(10)

    print(bboxes)
    print(f"bboxes B : =============> {len(bboxes)}")
    # merge boxes with iou > 0.1 as they are likely to be the same box
    bboxes = merge_boxes(bboxes, 0.08)
    t = timeit.timeit(lambda: merge_boxes(bboxes, 0.08), number=100) / 100

    print(f"t = {t}")

    # assert len(bboxes) == 1


def test_overlap_002():
    bboxes = generate_bboxes(10000)

    print(f"bboxes B : =============> {len(bboxes)}")
    # merge boxes with iou > 0.1 as they are likely to be the same box
    bboxes = merge_boxes(bboxes, 0.08)
    t = timeit.timeit(lambda: merge_boxes(bboxes, 0.08), number=1000) / 100

    print(f"t = {t}")

    # assert len(bboxes) == 1
