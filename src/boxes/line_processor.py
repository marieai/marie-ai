from __future__ import print_function

import copy
import os
from typing import List, Any

import cv2
import numpy as np

from utils.overlap import find_overlap


def find_line_index(lines, box):
    """Get line index for specific box"""

    _, line_indexes = find_overlap(box, lines)
    line_number = -1
    if len(line_indexes) == 1:
        line_number = line_indexes[0] + 1

    if line_number == -1:
        raise Exception(f"Invalid line number : -1, this looks like a bug : {line_indexes}, {box}")
    return line_number


def line_refiner(image, bboxes, _id, lines_dir) -> List[Any]:
    """Line refiner creates lines out of set of bounding box regions"""
    img_h = image.shape[0]
    img_w = image.shape[1]
    all_box_lines = []

    for idx, region in enumerate(bboxes):
        region = np.array(region).astype(np.int32).reshape((-1))
        region = region.reshape(-1, 2)
        poly = region.reshape((-1, 1, 2))
        box = cv2.boundingRect(poly)
        box = np.array(box).astype(np.int32)
        x, y, w, h = box
        box_line = [0, y, img_w, h]
        box_line = np.array(box_line).astype(np.int32)
        all_box_lines.append(box_line)
        # print(f' >  {idx} : {box} : {box_line}')
    # print(f'all_box_lines : {len(all_box_lines)}')

    all_box_lines = np.array(all_box_lines)
    if len(all_box_lines) == 0:
        return []

    y1 = all_box_lines[:, 1]

    # sort boxes by the  y-coordinate of the bounding box
    idxs = np.argsort(y1)
    lines = []
    size = len(idxs)
    iter_idx = 0

    while len(idxs) > 0:
        last = len(idxs) - 1
        idx = idxs[last]
        box_line = all_box_lines[idx]
        overlaps, indexes = find_overlap(box_line, all_box_lines)
        overlaps = np.array(overlaps)

        min_x = overlaps[:, 0].min()
        min_y = overlaps[:, 1].min()
        max_w = overlaps[:, 2].max()
        max_h = overlaps[:, 3].max()
        max_y = 0

        for overlap in overlaps:
            x, y, w, h = overlap
            dh = y + h
            if dh > max_y:
                max_y = dh

        max_h = max_y - min_y
        box = [min_x, min_y, max_w, max_h]
        lines.append(box)

        # there is a bug when there is a box index greater than candidate index
        # last/idx : 8   ,  2  >  [0 1 4 3 6 5 7 8 2] len = 9  /  [0 1 2 3 4 5 6 7 8 9] len = 10
        # Ex : 'index 9 is out of bounds for axis 0 with size 9'
        #  numpy.delete(arr, obj, axis=None)[source]¶
        indexes = indexes[indexes < idxs.size]
        idxs = np.delete(idxs, indexes, axis=0)
        iter_idx = iter_idx + 1
        # prevent inf loop
        if iter_idx > size:
            print("ERROR:Infinite loop detected")
            raise Exception("ERROR:Infinite loop detected")

    # reverse to get the right order
    lines = np.array(lines)[::-1]
    img_line = copy.deepcopy(image)

    for line in lines:
        x, y, w, h = line
        color = list(np.random.random(size=3) * 256)
        cv2.rectangle(img_line, (x, y), (x + w, y + h), color, 1)

    cv2.imwrite(os.path.join(lines_dir, "%s-line.png" % _id), img_line)

    # refine lines as there could be lines that overlap
    print(f"***** Line candidates size {len(lines)}")

    # sort boxes by the y-coordinate of the bounding box
    y1 = lines[:, 1]
    idxs = np.argsort(y1)
    refine_lines = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        idx = idxs[last]

        box_line = lines[idx]
        overlaps, indexes = find_overlap(box_line, lines)
        overlaps = np.array(overlaps)

        min_x = overlaps[:, 0].min()
        min_y = overlaps[:, 1].min()
        max_w = overlaps[:, 2].max()
        max_h = overlaps[:, 3].max()

        box = [min_x, min_y, max_w, max_h]
        refine_lines.append(box)

        # there is a bug when there is a box index greater than candidate index
        # last/idx : 8   ,  2  >  [0 1 4 3 6 5 7 8 2] len = 9  /  [0 1 2 3 4 5 6 7 8 9] len = 10
        # Ex : 'index 9 is out of bounds for axis 0 with size 9'
        #  numpy.delete(arr, obj, axis=None)[source]¶
        indexes = indexes[indexes < idxs.size]
        idxs = np.delete(idxs, indexes, axis=0)

    print(f"Final line size : {len(refine_lines)}")
    lines = np.array(refine_lines)[::-1]  # Reverse
    print(lines)

    img_line = copy.deepcopy(image)

    for line in lines:
        x, y, w, h = line
        color = list(np.random.random(size=3) * 256)
        cv2.rectangle(img_line, (x, y), (x + w, y + h), color, 1)

    cv2.imwrite(os.path.join(lines_dir, "%s-line.png" % (_id)), img_line)

    line_size = len(lines)
    print(f"Estimated line count : {line_size}")

    return lines
