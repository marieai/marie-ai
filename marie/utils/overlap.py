import numpy as np

import copy

# https://gist.github.com/YaYaB/39f9df9d481d784b786ad88eea8533e8


def find_overlap(box, data, overlap_ratio=0.75):
    """Find overlap between a box and a data set"""
    overlaps = []
    indexes = []

    if len(data) == 0:
        return np.array([]), np.array([])

    x, y, w, h = box
    x1min = x
    x1max = x + w
    y1min = y
    y1max = y + h
    # TODO, this needs to be configurable
    dyr = h * overlap_ratio

    # print(f'dyr = {dyr}')
    for i, bb in enumerate(data):
        _x, _y, _w, _h = bb
        x2min = _x
        x2max = _x + _w
        y2min = _y
        y2max = _y + _h

        if x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max:
            dy = min(y1max, y2max) - max(y1min, y2min)
            # print(f'ty : {dy} : {dyr}')
            if dy < dyr:
                continue

            overlaps.append(bb)
            indexes.append(i)

    return overlaps, indexes


def find_overlap_vertical(box, data, overlap_ratio=0.75, bidirectional: bool = True):
    """Find overlap between a box and a data set
    expected box format in [x, y, w, h]
    """
    overlaps = []
    indexes = []
    scores = []

    if len(data) == 0:
        return [], [], []

    # print("overlap ***********")
    # print(data)

    x, y, w, h = box
    y1min = y
    y1max = y + h
    # TODO, this needs to be configurable
    dyr = h * overlap_ratio
    # print(f'dyr = {dyr}')
    for i, bb in enumerate(data):
        _x, _y, _w, _h = bb
        y2min = _y
        y2max = _y + _h

        if h <= 0 or _h <= 0:
            continue

        # don't overlap exactly same boxes as target
        if np.array_equal(box, bb):
            continue

        y_bottom = min(y1max, y2max)
        y_top = max(y1min, y2min)

        if y1min < y2max and y2min < y1max:
            # intersection_area = min(y1max, y2max) - max(y1min, y2min)
            intersection_area = y_bottom - y_top
            # intersection_area = (x_right - x_left) * (y_bottom - y_top)
            intersection_area = y_bottom - y_top

            # compute the area of both AABBs
            # bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
            # bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
            # We are giving our box areas same width as we areonly interested in H IOU
            faux_w = 1
            bb1_area = faux_w * h
            bb2_area = faux_w * _h

            dr = h / _h
            # bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth areas - the interesection area
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
            # print(f"intersection_area  [{h} , {_h}]: {intersection_area} : {dr}  > iou = {iou}")
            assert iou >= 0.0
            assert iou <= 1.0

            scores.append(iou)
            overlaps.append(bb)
            indexes.append(i)

    return overlaps, indexes, scores


def find_overlap_horizontal(box, bboxes, center_y_overlap=None):
    """Find overlap between a box and a data set
    expected box format in [x, y, w, h]
    """

    overlaps = []
    indexes = []
    scores = []

    if len(bboxes) == 0:
        return [], [], []

    bboxes = np.array(bboxes)
    # filter out boxes that are not intersecting with the target box
    # intersecting_boxes = bboxes[
    #     (box[0] < bboxes[:, 0] + bboxes[:, 2]) & (box[0] + box[2] > bboxes[:, 0])
    # ]
    # intersecting_boxes = bboxes[box[0] < bboxes[:, 0] + bboxes[:, 2]]

    x, y, w, h = box
    x1min = x
    x1max = x + w

    center_start = 0
    center_end = 0

    if center_y_overlap is not None:
        center_start = (y + h // 2) - (h * center_y_overlap)
        center_end = (y + h // 2) + (h * center_y_overlap)

    for i, bb in enumerate(bboxes):
        _x, _y, _w, _h = bb
        x2min = _x
        x2max = _x + _w

        # don't overlap exactly same boxes as target
        if np.array_equal(box, bb):
            continue

        x_right = min(x1max, x2max)
        x_left = max(x1min, x2min)
        if x1min < x2max and x2min < x1max:
            # this is to make sure that the center of the box is within the center_y_overlap
            if center_y_overlap is not None:
                if _y + _h // 2 < center_start or _y + _h // 2 > center_end:
                    continue

            # intersection_area = min(y1max, y2max) - max(y1min, y2min)
            # intersection_area = (x_right - x_left) * (y_bottom - y_top)
            intersection_area = x_right - x_left

            # compute the area of both AABBs
            # bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
            # bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
            # We are giving our box areas same width as we are only interested in W IOU
            faux_h = 1
            bb1_area = faux_h * w
            bb2_area = faux_h * _w

            # bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth areas - the intersection area
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
            # print(f"intersection_area  [{h} , {_h}]: {intersection_area} : {dr}  > iou = {iou}")
            assert iou >= 0.0
            assert iou <= 1.0

            scores.append(iou)
            overlaps.append(bb)
            indexes.append(i)

    return overlaps, indexes, scores


def merge_bboxes_as_block(bboxes):
    """Merge bounding boxes into one block"""
    bboxes = np.array(bboxes)

    min_x = bboxes[:, 0].min()
    min_y = bboxes[:, 1].min()
    max_h = bboxes[:, 3].max()
    max_w = (bboxes[:, 0] + bboxes[:, 2]).max() - min_x
    block = [min_x, min_y, max_w, max_h]
    block = [round(k, 6) for k in block]

    return block


def compute_iou(box1, box2):
    """
    Compute the intersection over union of two set of boxes, each box is [x1, y1, x2, y2]
    @param box1:
    @param box2:
    @return:
    """

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # get the center of the box
    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))

    # print(
    #     f"y_overlap: {y_overlap} y_overlap2: {y_overlap2}  : box1: {box1} box2: {box2}"
    # )

    intersection = x_overlap * y_overlap
    area1 = (x2 - x1) * ((y2 - y1) // 4)  # box1 area
    area2 = (x4 - x3) * ((y4 - y3) // 4)  # box2 area
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou


def merge_boxes_by_iou(bboxes, iou_threshold: float = 0.5):
    """
    Merge boxes with iou > iou_threshold each box is [x1, y1, x2, y2]

    @param bboxes:
    @param iou_threshold:
    @return:
    """

    iou_threshold = 0.05
    merged_bboxes = []

    for box in bboxes:
        if len(merged_bboxes) == 0:
            merged_bboxes.append(box)
        else:
            merged = False
            for merged_box in merged_bboxes:
                iou = compute_iou(box, merged_box)
                if iou > 0.0:
                    print(f"iou: {iou}  : box: {box}  : merged_box: {merged_box}")

                if iou > iou_threshold:
                    merged_box[0] = min(box[0], merged_box[0])
                    merged_box[1] = min(box[1], merged_box[1])
                    merged_box[2] = max(box[2], merged_box[2])
                    merged_box[3] = max(box[3], merged_box[3])
                    merged = True
                    break
            if not merged:
                merged_bboxes.append(box)
    return merged_bboxes


def merge_boxes(bboxes_xyxy, delta_x=0.0, delta_y=0.0):
    """
    Merge boxes that are close to each other and have center y overlap
    @param bboxes_xyxy:
    @param delta_x:
    @param delta_y:
    @return:
    """

    # return bboxes_xyxy
    # convert to [x, y, w, h]
    bboxes = []
    for bbox in bboxes_xyxy:
        bb = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        bboxes.append(bb)

    # bboxes = sorted(bboxes, key=lambda x: x[1])
    last_box_size = len(bboxes)
    max_consecutive_merges = 3

    while max_consecutive_merges > 0:
        visited = [False for _ in range(0, len(bboxes))]
        bboxes_to_merge = {}
        for idx in range(0, len(bboxes)):
            if visited[idx]:
                continue
            visited[idx] = True
            box = bboxes[idx]

            overlaps, indexes, scores = find_overlap_horizontal(
                box, bboxes, center_y_overlap=0.5
            )

            bboxes_to_merge[idx] = [idx]
            for _, overlap_idx, score in zip(overlaps, indexes, scores):
                # print("overlap_idx", overlap_idx, "scores", scores)
                visited[overlap_idx] = True
                bboxes_to_merge[idx].append(overlap_idx)

        if len(bboxes_to_merge) == len(bboxes):
            break

        new_blocks = []
        for _k, idxs in bboxes_to_merge.items():
            items = np.array(bboxes)
            picks = items[idxs]
            block = merge_bboxes_as_block(picks)
            new_blocks.append(block)

        bboxes = new_blocks

        if last_box_size == len(bboxes):
            break

        max_consecutive_merges -= 1
        last_box_size = len(bboxes)

    # convert to [x1, y1, x2, y2] format for output
    bboxes_merged_xyxy = []
    for bbox in bboxes:
        block_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bboxes_merged_xyxy.append(block_xyxy)
    return bboxes_merged_xyxy
