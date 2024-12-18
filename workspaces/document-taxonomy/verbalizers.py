import copy
from math import ceil
from typing import Dict, List

import numpy as np


def verbalizers_PLAIN_TEXT(metadata):
    """
    Plain text verbalizer, no spatial context
    """
    return copy.deepcopy(metadata["lines"])


def verbalizers_LMDX(metadata):
    """
    lmdx: Language Model-based Document Information Extraction and Localization
    https://arxiv.org/pdf/2309.10952
    """

    lines = copy.deepcopy(metadata["lines"])
    for line in lines:
        line_text = line["text"]
        line_number = line["line"]
        line_bbox_xywh = [int(x) for x in line["bbox"]]
        x, y = line_bbox_xywh[0], line_bbox_xywh[1]
        line["text"] = f"{line_text} {x}|{y}"
    return lines


def verbalizers_SPATIAL_FORMAT(metadata):
    """
    SpatialFormat Uses the geometries to restore the original document layout via insertion of spaces and newlines.
    To this end, the characters are placed on a grid such that their spatial location is similar to that on the document
    https://arxiv.org/pdf/2402.09841v1
    """
    lines = copy.deepcopy(metadata["lines"])
    words = metadata["words"]

    meta = metadata["meta"]["imageSize"]
    width = meta["width"]
    height = meta["height"]

    char_width = 8.44
    cols = ceil(width // char_width)

    x_space = np.arange(0, width, 1)
    bins = np.linspace(0, width, cols)
    bins = np.array(bins).astype(np.int32)
    x_hist = np.digitize(x_space, bins, right=True)
    max_characters_per_line = ceil(width // char_width)

    for line in lines:
        wordids = line[
            "wordids"
        ]  # this are Word ID not Indexes, each word is assigned a unique ID when it is created
        line_bbox_xywh = [int(x) for x in line["bbox"]]
        x, y, w, h = line_bbox_xywh
        aligned_words = [w for w in words if w["id"] in wordids]
        last_space = 0
        line_buffer = " " * max_characters_per_line
        SPACES = 4

        for idx, word in enumerate(aligned_words):
            curr_box = aligned_words[idx]["box"]
            text = word["text"]
            x2, y2, w2, h2 = curr_box
            grid_space = x_hist[x2]
            spaces = max(
                1, (grid_space - last_space)
            )  # Compress each four spaces to one
            last_space = grid_space + len(
                text
            )  # Update last_space to the end of the current word
            line_buffer = (
                line_buffer[:last_space] + text + line_buffer[last_space + len(text) :]
            )
            # print(f"{grid_space} : {spaces}  > {text}")

        line_buffer = line_buffer.replace(" " * SPACES, " ")
        line_buffer = line_buffer.rstrip()
        print(line_buffer)

        x, y = line_bbox_xywh[0], line_bbox_xywh[1]
        verbalizer = f"{line_buffer} {x}|{y}"
        # line["text"] = verbalizer
        line["text"] = line_buffer
    return lines


def verbalizers(method: str, metadata) -> List[Dict]:
    method = method.upper()
    if method == "LMDX":
        return verbalizers_LMDX(metadata)
    elif method == "PLAIN_TEXT":
        return verbalizers_PLAIN_TEXT(metadata)
    elif method == "SPATIAL_FORMAT":
        return verbalizers_SPATIAL_FORMAT(metadata)
    else:
        raise ValueError(f"Unknown method: {method}")
