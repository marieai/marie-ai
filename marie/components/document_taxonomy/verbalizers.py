import copy
from math import ceil
from typing import Dict, List, Optional

import numpy as np

from marie.utils.overlap import merge_bboxes_as_block


def create_chunks(metadata, tokenizer, max_token_length: Optional[int]) -> List[Dict]:
    """
    Divides a document into chunks based on max token length.
    For an LLM, the context window should "have text around" the target point, meaning it should include both text before and after the current point of focus (the line)
    """
    chunks = []
    if max_token_length is None:
        max_token_length = tokenizer.model_max_length

    lines = verbalizers("LMDX", metadata)
    meta_size = metadata["meta"]["imageSize"]
    w, h = meta_size["width"], meta_size["height"]

    for idx, line in enumerate(lines):
        line_id = line["line"]
        line_bbox_xywh = [int(x) for x in line["bbox"]]
        chunk_size_start = 0
        chunk_size_end = 0
        chunk_idx = 0
        token_length = 0
        prompt = ""
        q = ""
        c = ""
        collected_bbox_xywh = []

        while token_length <= max_token_length:
            start = max(0, idx - chunk_size_start)
            end = min(len(lines), idx + chunk_size_end)
            source_row = lines[idx]
            selected_rows = lines[start:end]

            q = source_row["text"]
            c = "\n".join([r["text"] for r in selected_rows])
            current_prompt = f"""classify: {q}\ncontext: {c}\n"""
            collected_bbox_xywh = [line["bbox"] for line in selected_rows]
            tokens = tokenizer(
                current_prompt, return_tensors="pt", add_special_tokens=False
            )
            token_count = len(tokens["input_ids"][0])
            if token_count > max_token_length:
                break

            if chunk_idx % 2 == 0:
                chunk_size_start += 1
            else:
                chunk_size_end += 1

            chunk_idx += 1
            prompt = current_prompt
            token_length = token_count

            if start == 0 and end == len(lines):
                break

        block_xywh = merge_bboxes_as_block(collected_bbox_xywh)

        block_xywh[0] = 0
        block_xywh[2] = w

        chunks.append(
            {
                "line_id": line_id,
                "question": q,
                "context": c,
                "bbox": block_xywh,
                "question_bbox": line_bbox_xywh,
                "prompt": prompt,
            }
        )
    return chunks


def group_taxonomies_by_label(lines: List[Dict]) -> List[Dict]:
    """
    Groups contiguous lines with the same label into taxonomy groups.
    """
    if len(lines) == 0:
        return []

    grouped_lines = []
    current_group = {"label": lines[0]["taxonomy"]["label"], "lines": [lines[0]]}

    for line in lines[1:]:
        if line["taxonomy"]["label"] == current_group["label"]:
            current_group["lines"].append(line)
        else:
            grouped_lines.append(current_group)
            current_group = {"label": line["taxonomy"]["label"], "lines": [line]}

    grouped_lines.append(current_group)  # Add the last group

    for group in grouped_lines:
        print(f"Group: {group['label']}")
        group_size = len(group["lines"])
        total_score = 0
        min_x, min_y, max_x, max_y = (
            float('inf'),
            float('inf'),
            float('-inf'),
            float('-inf'),
        )
        for line in group["lines"]:
            score = line['taxonomy']['score']
            total_score += score
            score = f"{score:.4f}"
            line_info = f"Line {line['line']}: {score} > {line['text']}"
            print(line_info)
            bbox = line['bbox']
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[0] + bbox[2])
            max_y = max(max_y, bbox[1] + bbox[3])
        average_score = total_score / group_size
        print(f"Average Score for Group '{group['label']}': {average_score:.4f}")
        group['bbox'] = [min_x, min_y, max_x - min_x, max_y - min_y]
        group['score'] = average_score
        print(f"Bounding Box for Group '{group['label']}': {group['bbox']}")

    return grouped_lines


def verbalizers_PLAIN_TEXT(metadata):
    """
    Plain text verbalizer, no spatial context
    """
    return copy.deepcopy(metadata["lines"])


def _quantize(values, n=99):
    """
    Takes a list of values and returns the returns a list of buckets
    after quantizing the values to n buckets
    """
    if not values:
        return []  # Return an empty list if input is empty

    min_value = min(values)
    max_value = max(values)
    bucket_width = (max_value - min_value) / n
    bucket_indices = []
    for value in values:
        index = int(
            (value - min_value) / bucket_width
        )  # Determine which bucket it belongs to
        bucket_indices.append(index)
    return bucket_indices


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def normalize_and_quantize(metadata):
    """
    Normalize and quantize the bounding boxes in the metadata
    """

    meta = metadata["meta"]
    image_width = meta["imageSize"]["width"]
    image_height = meta["imageSize"]["height"]
    if not image_width or not image_height:
        raise ValueError("Width and height must be provided in metadata")

    lines = copy.deepcopy(metadata["lines"])
    # turn coordinates into (left, top, left+width, top+height) format (x1, y1, x2, y2)
    actual_boxes = []
    for line in lines:
        x, y, w, h = [int(x) for x in line["bbox"]]
        actual_box = [x, y, x + w, y + h]
        actual_boxes.append(actual_box)

    # finally, normalize the bounding boxes
    normalized_boxes = []
    for box in actual_boxes:
        normalized_box = normalize_box(box, image_width, image_height)
        x_mid = (normalized_box[0] + normalized_box[2]) / 2
        y_mid = (normalized_box[1] + normalized_box[3]) / 2
        normalized_boxes.append([x_mid, y_mid])

    print(normalized_boxes)

    xs = _quantize([int(x[0]) for x in normalized_boxes])
    ys = _quantize([int(x[1]) for x in normalized_boxes])

    return xs, ys, normalized_boxes


def verbalizers_LMDX(metadata):
    """
    lmdx: Language Model-based Document Information Extraction and Localization
    https://arxiv.org/pdf/2309.10952
    """
    xs, ys, normalized_boxes = normalize_and_quantize(metadata)
    index = 0
    lines = copy.deepcopy(metadata["lines"])

    for line, qx, qy in zip(lines, xs, ys):
        line_text = line["text"]
        line_number = line["line"]
        line_bbox_xywh = [int(x) for x in line["bbox"]]
        # x,y = line_bbox_xywh[0],line_bbox_xywh[1]
        line["text"] = f"{line_text} {qx:02d}|{qy:02d}"
        index += 1
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

    lines = copy.deepcopy(metadata["lines"])
    xs, ys, normalized_boxes = normalize_and_quantize(metadata)
    index = 0

    for line, qx, qy in zip(lines, xs, ys):
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
        line["text"] = f"<{index}> {line_buffer} {qx:02d}|{qy:02d}"
        index += 1
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
