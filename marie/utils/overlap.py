import numpy as np


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
    """Find overlap between a box and a data set"""
    overlaps = []
    indexes = []
    scores = []

    if len(data) == 0:
        return [], [], []

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
