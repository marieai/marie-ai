import copy
from math import ceil
from typing import Dict, List, Optional

import numpy as np

from marie.logging_core.predefined import default_logger as logger
from marie.utils.overlap import merge_bboxes_as_block


def create_chunks(
    metadata, tokenizer, max_token_length: Optional[int], method='LMDX', mode='line'
) -> List[Dict]:
    """
    Divides a document into chunks based on max token length.
    For an Classification LLM, the context window should "have text around" the target point, meaning it should include both text before and after the current point of focus (the line)
    For an Seq2Seq LLM, the context is the lines to classify
    """
    chunks = []
    if max_token_length is None:
        max_token_length = tokenizer.model_max_length

    lines = verbalizers(method, metadata)
    meta_size = metadata["meta"]["imageSize"]
    w, h = meta_size["width"], meta_size["height"]

    max_lines_per_chunk = 16
    print(f"Selected max_lines_per_chunk: {max_lines_per_chunk}")
    for idx, line in enumerate(lines):
        line_id = line["line"]
        line_text = line["text"]
        line_bbox_xywh = [int(x) for x in line["bbox"]]
        chunk_size_start = 0
        chunk_size_end = 0
        chunk_idx = 0
        token_length = 0
        prompt = ""
        last_q = ""
        last_c = ""
        collected_bbox_xywh = []

        if line_text == "":
            continue

        # https://github.com/google-research/FLAN/blob/main/flan/v2/flan_templates_branched.py
        target_rows = []
        max_iterations = 100  # Limit to prevent infinite loop
        iterations = 0

        while token_length <= max_token_length:
            if iterations >= max_iterations:
                print("Breaking loop due to max_iterations limit")
                raise ValueError("Breaking loop due to max_iterations limit")
                # break  # Safeguard against an infinite loop

            iterations += 1
            start = max(0, idx - chunk_size_start)
            end = min(len(lines), idx + chunk_size_end)
            source_row = lines[idx]
            selected_rows = lines[start:end]

            if len(target_rows) < max_lines_per_chunk:
                target_rows = selected_rows[:max_lines_per_chunk]

            if mode == 'seq2seq':
                q = "\n".join([r["text"] for r in target_rows])
                c = "\n".join([r["text"] for r in selected_rows])
                current_prompt = f"""classify each row:\n\n{q}\n\ncontext:\n\n{c}\n\nOPTIONS:\n-TABLE \n-SECTION \n-CODE \n-OTHER"""
                current_prompt = f"""classify each row:\n\n{q}\n\nOPTIONS:\n-TABLE \n-SECTION \n-CODE \n-OTHER"""
            else:
                q = source_row["text"]
                c = "\n".join([r["text"] for r in selected_rows])
                current_prompt = f"""classify: {q}\ncontext: {c}\n"""

            collected_bbox_xywh = [line["bbox"] for line in selected_rows]
            tokens = tokenizer(
                current_prompt, return_tensors="pt", add_special_tokens=False
            )
            current_length = len(tokens["input_ids"][0])
            if current_length > max_token_length:
                break

            if mode == 'seq2seq':
                chunk_size_start = 0
                chunk_size_end += 1
            else:
                if chunk_idx % 2 == 0:
                    chunk_size_start += 1
                else:
                    chunk_size_end += 1

            chunk_idx += 1
            prompt = current_prompt
            token_length = current_length
            last_q = q
            last_c = c

            if start == 0 and end == len(lines) or (start == idx and end == len(lines)):
                break

            if max_lines_per_chunk == len(target_rows):
                break

        block_xywh = merge_bboxes_as_block(collected_bbox_xywh)

        block_xywh[0] = 0
        block_xywh[2] = w
        line_ids = [r["line"] for r in target_rows]

        chunks.append(
            {
                "line_id": line_id,
                "question": last_q,
                "context": last_c,
                "bbox": block_xywh,
                "question_bbox": line_bbox_xywh,
                "prompt": prompt,
                "line_ids": line_ids,
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
    min_value = min(values)
    max_value = max(values)
    bucket_width = (max_value - min_value) / n
    bucket_indices = []

    if bucket_width == 0:  # Avoid division by zero
        bucket_width = 1

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


def normalize_and_quantize(metadata: Dict) -> tuple:
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

    # normalize the bounding boxes
    normalized_boxes = []
    for box in actual_boxes:
        normalized_box = normalize_box(box, image_width, image_height)
        x_mid = (normalized_box[0] + normalized_box[2]) / 2
        y_mid = (normalized_box[1] + normalized_box[3]) / 2
        normalized_boxes.append([x_mid, y_mid])

    if len(normalized_boxes) == 0:
        return [], [], []

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


def verbalizers_SPATIAL_FORMAT(metadata, min_spacing=2, target_columns=160):
    """
    Aligns words using quantized grid with:
    - Dynamic char width
    - Global column compression
    - Whole-word block placement
    """
    words = metadata["words"]
    meta = metadata["meta"]["imageSize"]
    width = meta["width"]

    # character width estimation
    word_widths = [w["box"][2] for w in words if len(w["text"]) > 0]
    word_lengths = [len(w["text"]) for w in words if len(w["text"]) > 0]

    if word_lengths:
        avg_char_width = np.mean(word_widths) / np.mean(word_lengths)
    else:
        avg_char_width = width / target_columns  # fallback

    char_width = max(4.0, min(avg_char_width, 10.0))
    max_chars_per_line = ceil(width / char_width)
    cols = max_chars_per_line

    x_space = np.arange(0, width, 1)
    bins = np.linspace(0, width, cols).astype(np.int32)
    x_hist = np.digitize(x_space, bins, right=True)

    lines = copy.deepcopy(metadata["lines"])
    xs, ys, normalized_boxes = normalize_and_quantize(metadata)

    if len(normalized_boxes) == 0:
        logger.warning("No normalized boxes found in metadata.")
        print('metadata', metadata)
        return []

    all_word_positions = []  # Stores [(line_index, grid_index, word)]

    for line_idx, (line, qx, qy, normal_box) in enumerate(
        zip(lines, xs, ys, normalized_boxes)
    ):
        wordids = line["wordids"]
        aligned_words = [w for w in words if w["id"] in wordids]

        for word in aligned_words:
            x, y, w, h = word["box"]
            text = word["text"]

            x_clamped = min(x, len(x_hist) - 1)
            grid_index = x_hist[x_clamped]
            all_word_positions.append((line_idx, grid_index, text))

    # Column usage
    column_usage = [0] * max_chars_per_line
    for _, grid_index, _ in all_word_positions:
        if grid_index < max_chars_per_line:
            column_usage[grid_index] += 1

    active_columns = [i for i, count in enumerate(column_usage) if count > 0]

    # column compression map
    compressed_column_map = {}
    new_idx = 0
    for i, col in enumerate(active_columns):
        compressed_column_map[col] = new_idx
        next_col = (
            active_columns[i + 1] if i + 1 < len(active_columns) else col + min_spacing
        )
        gap = max(min_spacing, next_col - col)
        new_idx += gap

    final_line_length = max(compressed_column_map.values()) + 40  # some breathing room

    # compressed lines
    final_buffers = [[" "] * final_line_length for _ in range(len(lines))]

    for line_idx, grid_index, text in all_word_positions:
        if grid_index in compressed_column_map:
            insert_pos = compressed_column_map[grid_index]
            if insert_pos + len(text) <= final_line_length:
                target_buffer = final_buffers[line_idx]
                slice_ = target_buffer[insert_pos : insert_pos + len(text)]
                if all(c == " " for c in slice_):  # no overlap
                    for offset, char in enumerate(text):
                        target_buffer[insert_pos + offset] = char

    for idx, line in enumerate(lines):
        line["text"] = "".join(final_buffers[idx]).rstrip()

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
