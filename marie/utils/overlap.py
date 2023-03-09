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


def find_overlap_horizontal(box, bboxes):
    """Find overlap between a box and a data set
    expected box format in [x, y, w, h]
    """

    overlaps = []
    indexes = []
    scores = []

    if len(bboxes) == 0:
        return [], [], []

    x, y, w, h = box
    x1min = x
    x1max = x + w

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
            # area and dividing it by the sum of prediction + ground-truth areas - the interesection area
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


def merge_boxes(bboxes, delta_x=0.0, delta_y=0.0):
    """
    Arguments:
        bboxes {list} -- list of bounding boxes with each bounding box is a list [xmin, ymin, xmax, ymax]
        delta_x {float} -- margin taken in width to merge
        detlta_y {float} -- margin taken in height to merge
    Returns:
        {list} -- list of bounding boxes merged

    https://gist.github.com/YaYaB/39f9df9d481d784b786ad88eea8533e8
    """

    def is_in_bbox(point, bbox):
        """
        Arguments:
            point {list} -- list of float values (x,y)
            bbox {list} -- bounding box of float_values [xmin, ymin, xmax, ymax]
        Returns:
            {boolean} -- true if the point is inside the bbox
        """
        return (
            point[0] >= bbox[0]
            and point[0] <= bbox[2]
            and point[1] >= bbox[1]
            and point[1] <= bbox[3]
        )

    def intersect(bbox, bbox_):
        """
        Arguments:
            bbox {list} -- bounding box of float_values [xmin, ymin, xmax, ymax]
            bbox_ {list} -- bounding box of float_values [xmin, ymin, xmax, ymax]
        Returns:
            {boolean} -- true if the bboxes intersect
        """
        for i in range(int(len(bbox) / 2)):
            for j in range(int(len(bbox) / 2)):
                # Check if one of the corner of bbox inside bbox_
                if is_in_bbox([bbox[2 * i], bbox[2 * j + 1]], bbox_):
                    return True
        return False

    def intersectXX(bbox, bbox_):
        return (
            bbox[0] < bbox_[2]
            and bbox[2] > bbox_[0]
            and bbox[1] < bbox_[3]
            and bbox[3] > bbox_[1]
        )

    # Sort bboxes by ymin
    bboxes = sorted(bboxes, key=lambda x: x[1])

    tmp_bbox = None
    while True:
        nb_merge = 0
        used = []
        new_bboxes = []
        # Loop over bboxes
        for i, b in enumerate(bboxes):
            for j, b_ in enumerate(bboxes):
                # If the bbox has already been used just continue
                if i in used or j <= i:
                    continue
                # Compute the bboxes with a margin
                bmargin = [
                    b[0] - (b[2] - b[0]) * delta_x,
                    b[1] - (b[3] - b[1]) * delta_y,
                    b[2] + (b[2] - b[0]) * delta_x,
                    b[3] + (b[3] - b[1]) * delta_y,
                ]
                b_margin = [
                    b_[0] - (b_[2] - b_[0]) * delta_x,
                    b_[1] - (b[3] - b[1]) * delta_y,
                    b_[2] + (b_[2] - b_[0]) * delta_x,
                    b_[3] + (b_[3] - b_[1]) * delta_y,
                ]
                # Merge bboxes if bboxes with margin have an intersection
                # Check if one of the corner is in the other bbox
                # We must verify the other side away in case one bounding box is inside the other
                if intersect(bmargin, b_margin) or intersect(b_margin, bmargin):
                    tmp_bbox = [
                        min(b[0], b_[0]),
                        min(b[1], b_[1]),
                        max(b_[2], b[2]),
                        max(b[3], b_[3]),
                    ]
                    used.append(j)
                    # print(bmargin, b_margin, 'done')
                    nb_merge += 1
                if tmp_bbox:
                    b = tmp_bbox
            if tmp_bbox:
                new_bboxes.append(tmp_bbox)
            elif i not in used:
                new_bboxes.append(b)
            used.append(i)
            tmp_bbox = None
        # If no merge were done, that means all bboxes were already merged
        if nb_merge == 0:
            break
        bboxes = copy.deepcopy(new_bboxes)

    return new_bboxes


def merge_boxesZZZ(bboxes, iou_threshold: float = 0.5):
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

                # get center of box
                x1, y1, x2, y2 = box
                x3, y3, x4, y4 = merged_box

                y_center_1 = y1 + ((y2 - y1) // 2)

                # # check if the center of the box is within the merged box 20 pixels
                # if not (y_center_1 > y3 and y_center_1 < y4):
                #     continue

                iou = compute_iou(box, merged_box)
                if iou > 0.0:
                    print(f"iou: {iou}  : box: {box}  : merged_box: {merged_box}")

                if iou > 0:
                    merged_box[0] = min(box[0], merged_box[0])
                    merged_box[1] = min(box[1], merged_box[1])
                    merged_box[2] = max(box[2], merged_box[2])
                    merged_box[3] = max(box[3], merged_box[3])
                    merged = True
                    break
            if not merged:
                merged_bboxes.append(box)
    return merged_bboxes
