"""
This import path is important to allow importing correctly as package
"""

from __future__ import absolute_import

import os
import sys
from importlib import import_module

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

_PROCESSORS = {
    "AwsTextractOcrProcessor": ".aws_textract_ocr",
    "AzureVisionOcrProcessor": ".azure_vision_ocr",
    "CraftOcrProcessor": ".craft_ocr_processor",
    "GoogleVisionOcrProcessor": ".google_vision_ocr",
    "LevenshteinOcrProcessor": ".lev_ocr_processor",
    "TesseractOcrProcessor": ".tesseract_ocr_processor",
    "TrOcrProcessor": ".trocr_ocr_processor",
}

__all__ = [
    "CraftOcrProcessor",
    "TrOcrProcessor",
    "TesseractOcrProcessor",
    "LevenshteinOcrProcessor",
    "AwsTextractOcrProcessor",
    "GoogleVisionOcrProcessor",
    "AzureVisionOcrProcessor",
]  # noqa


def __getattr__(name):
    if name not in _PROCESSORS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_PROCESSORS[name], __name__)
    processor = getattr(module, name)
    globals()[name] = processor
    return processor
