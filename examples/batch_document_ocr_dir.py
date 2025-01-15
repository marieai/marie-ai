from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

from examples.utils import parse_args
from marie.ocr import DefaultOcrEngine, MockOcrEngine
from marie.utils.docs import frames_from_file
from marie.utils.json import store_json_object

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()

mock_ocr = False
ocr_engine = (
    MockOcrEngine(cuda=use_cuda) if mock_ocr else DefaultOcrEngine(cuda=use_cuda)
)


def process_request(mode: str, file_location: str) -> Any | None:
    """
    Processes a single file to extract OCR results.

    :param mode: The OCR mode (e.g., multiline, singleline, etc.).
    :param file_location: Path to the file to be processed.
    :return: OCR results or None if an error occurs.
    """
    if not os.path.exists(file_location):
        raise FileNotFoundError(f"File not found: {file_location}")

    try:
        logger.info(f"Processing file: {file_location}")
        frames = frames_from_file(file_location)
        return ocr_engine.extract(frames)
    except Exception as e:
        logger.error(f"Error processing file {file_location}: {e}")
        return None


def process_dir(src_dir: str, output_dir: str) -> None:
    """
    Processes a directory of images, extracts OCR results, and saves them as JSON.

    :param src_dir: Directory containing the source images.
    :param output_dir: Directory to save the OCR results.
    """
    src_dir = os.path.expanduser(src_dir)
    output_dir = os.path.expanduser(output_dir)

    for img_path in Path(src_dir).rglob("*"):
        if not img_path.is_file():
            continue

        resolved_output_path = os.path.join(output_dir, img_path.relative_to(src_dir))
        json_output_path = os.path.splitext(resolved_output_path)[0] + ".json"

        # Skip unsupported extensions
        file_extension = img_path.suffix.lower()
        if file_extension not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
            logger.warning(f"Skipping unsupported file: {img_path} ({file_extension})")
            continue

        if os.path.exists(json_output_path):
            logger.info(f"Skipping already processed file: {img_path}")
            continue

        os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)

        logger.info(f"Processing file: {img_path}")
        json_result = process_request(mode="multiline", file_location=str(img_path))

        if json_result is not None:
            store_json_object(json_result, json_output_path)
            logger.info(f"Saved OCR results to {json_output_path}")
        else:
            logger.error(f"Failed to process file: {img_path}")


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")

    process_dir(src_dir=args.input, output_dir=args.output_dir)
