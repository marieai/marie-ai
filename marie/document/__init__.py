"""
This import path is important to allow importing correctly as package
"""

from __future__ import absolute_import

import os
import sys

from .craft_ocr_processor import CraftOcrProcessor
from .trocr_ocr_processor import TrOcrProcessor
from .tesseract_ocr_processor import TesseractOcrProcessor
from .lev_ocr_processor import LevenshteinOcrProcessor
from .aws_textract_ocr import AwsTextractOcrProcessor
from .google_vision_ocr import GoogleVisionOcrProcessor
from .azure_vision_ocr import AzureVisionOcrProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

__all__ = [
    "CraftOcrProcessor",
    "TrOcrProcessor",
    "TesseractOcrProcessor",
    "LevenshteinOcrProcessor",
    "AwsTextractOcrProcessor",
    "GoogleVisionOcrProcessor",
    "AzureVisionOcrProcessor",
]  # noqa
