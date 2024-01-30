import os
import tempfile
from os import PathLike
from typing import Union

import numpy as np

from marie.renderer import TextRenderer
from marie.utils.json import load_json_file


def get_words_and_boxes(ocr_results, page_index: int) -> tuple:
    """
    Get words and boxes from OCR results.
    :param ocr_results:
    :param page_index:
    :return:
    """
    words = []
    boxes = []

    if not ocr_results:
        return words, boxes

    if page_index >= len(ocr_results):
        raise ValueError(f"Page index {page_index} is out of range.")

    for w in ocr_results[page_index]["words"]:
        boxes.append(w["box"])
        words.append(w["text"])
    return words, boxes


def meta_to_text(
    meta_or_path: Union[dict | str | PathLike], text_output_path: str = None
) -> str:
    """
    Convert meta data to text.

    :param meta_or_path: Meta data or path to meta data.
    :param text_output_path:  Path to text output file. If not provided, a temporary file will be used.
    :return:
    """

    if isinstance(meta_or_path, (str, PathLike)):
        results = load_json_file(meta_or_path)
    else:
        results = meta_or_path

    # create a fake frames array from metadata in the results, this is needed for the renderer for sizing
    frames = []

    for result in results:
        meta = result["meta"]["imageSize"]
        width = meta["width"]
        height = meta["height"]
        frames.append(np.zeros((height, width, 3), dtype=np.uint8))

    # write to temp file and read it back
    if text_output_path:
        tmp_file = open(text_output_path, "w")
    else:
        tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")

    with open(tmp_file.name, "w", encoding="utf-8") as f:
        for result in results:
            lines = result["lines"]
            lines = sorted(lines, key=lambda k: k["line"])
            for line in lines:
                f.write(line["text"] + "\n")

    if False:
        renderer = TextRenderer(config={"preserve_interword_spaces": False})
        renderer.render(
            frames,
            results,
            output_file_or_dir=tmp_file.name,
        )

    tmp_file.close()

    with open(tmp_file.name, "r") as f:
        return f.read()
