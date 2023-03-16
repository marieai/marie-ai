from __future__ import print_function

import os
from typing import Any, List

import cv2
import numpy as np

from marie.utils.overlap import find_overlap_vertical


def find_line_number(lines, box):
    """Get line index for specific box
    Args:
        lines: all lines to check against in (x, y, w, h) format
        box: box to check in (x, y, w, h) format
    Returns:
          line_number or -1 if line was not determined
    """
    line_number = -1
    overlaps, indexes, scores = find_overlap_vertical(box, lines)
    if len(indexes) == 1:
        line_number = indexes[0] + 1
    elif len(indexes) > 1:
        iou_best = 0
        for k, (overlap, index, score) in enumerate(zip(overlaps, indexes, scores)):
            if score > iou_best:
                iou_best = score
                line_number = index + 1

    if line_number == -1:
        msg = f"Invalid line number : -1, this looks like a bug/vertical line : {box}"
        print(msg)
        # raise Exception(msg)
        # find the closest line
        min_y = 100000
        for i, line in enumerate(lines):
            line_y = line[1] + line[3]
            box_y = box[1] + box[3] // 2
            dy = abs(box_y - line_y)
            if dy < min_y:
                line_number = i + 1
                min_y = dy

        print(f"Adjusted closest line_number = {line_number}")

    return line_number


def __line_merge(image, bboxes, min_iou=0.5) -> List[Any]:
    if len(bboxes) == 0:
        return []
    bboxes = np.array(bboxes)
    # sort boxes by the  y-coordinate of the bounding box
    y1 = bboxes[:, 1]
    idxs = np.argsort(y1)
    bboxes = bboxes[idxs]
    lines = []
    visited = [False for k in range(0, len(bboxes))]

    for idx in range(0, len(bboxes)):
        if visited[idx]:
            continue
        visited[idx] = True
        box = bboxes[idx]
        overlaps, indexes, scores = find_overlap_vertical(box, bboxes)
        # print(f" ***   {box}  -> : {len(overlaps)} ::: {overlaps} , {scores} , {indexes}")

        # now we check each overlap against each other
        # for each item that overlaps our box check to make sure that the ray back is valid
        exp_count = len(overlaps)
        idx_to_merge = [idx]

        for k, (overlap, index, score) in enumerate(zip(overlaps, indexes, scores)):
            # print(f"\t\t{k} -> {score} : {index} : {overlap}")
            if visited[index] or score < min_iou:
                continue
            bi_overlaps, bi_indexes, bi_scores = find_overlap_vertical(overlap, bboxes)
            if len(bi_overlaps) == exp_count:
                idx_to_merge.append(index)
                visited[index] = True
                s_h = box[3]
                # check if we have candidates that are fully overlapping the source
                if False:
                    for m, (bi_overlap, bi_index, bi_score) in enumerate(
                        zip(bi_overlaps, bi_indexes, bi_scores)
                    ):
                        c_h = bboxes[bi_index][3]
                        if c_h < s_h:
                            print(f"REMOVE : {bboxes[index]} :: {bboxes[bi_index]}")
                        # print(f"\t\t\t\t{k} -> {s_h}   === {c_h} ")
        lines.append(idx_to_merge)
        # print("*************")
        # print(f"idxs = {len(idx_to_merge)}")

    lines_bboxes = []

    for i, indexes in enumerate(lines):
        overlaps = bboxes[indexes]
        min_x = overlaps[:, 0].min()
        min_y = overlaps[:, 1].min()
        max_h = overlaps[:, 3].max()
        max_w = (overlaps[:, 0] + overlaps[:, 2]).max() - min_x
        box = [min_x, min_y, max_w, max_h]
        lines_bboxes.append(box)

    return lines_bboxes


def line_merge(image, bboxes) -> List[Any]:
    if len(bboxes) == 0:
        return []
    if image is not None:
        _h = image.shape[0]
        _w = image.shape[1]
    else:
        _h = 0
        _w = 0

    iou_scores = [0.6, 0.5, 0.4, 0.3]
    # iou_scores = [0.6, 0.6, 0.5, 0.35]
    merged_bboxes = bboxes
    for i in range(0, len(iou_scores)):
        # overlay = copy.deepcopy(image)
        merged_bboxes = __line_merge(image, merged_bboxes, iou_scores[i])
        if False:
            overlay = np.ones((_h, _w, 3), dtype=np.uint8) * 255
            for box in merged_bboxes:
                x, y, w, h = box
                color = (255, 0, 0)  # list(np.random.random(size=3) * 256)
                color = list(np.random.random(size=3) * 256)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
            cv2.imwrite(
                os.path.join("/tmp/fragments", f"overlay_refiner-{i}.png"), overlay
            )

    # sort boxes by the  y-coordinate of the bounding box
    # run final pass merge fully overlapping boxes
    idx_to_remove = []
    for i, box0 in enumerate(merged_bboxes):
        x0, y0, w0, h0 = box0
        for j, box1 in enumerate(merged_bboxes):
            if i == j:
                continue
            x1, y1, w1, h1 = box1
            if (
                ((x1 > x0) and (x1 + w1) < (x0 + w0))
                and (y1 > y0)
                and (y1 + h1) < (y0 + h0)
            ):
                idx_to_remove.append(j)

    merged_bboxes = np.array(merged_bboxes)
    if len(idx_to_remove) > 0:
        merged_bboxes = np.delete(merged_bboxes, np.unique(idx_to_remove), axis=0)

    y1 = merged_bboxes[:, 1]
    idxs = np.argsort(y1)
    merged_bboxes = merged_bboxes[idxs]

    return merged_bboxes
