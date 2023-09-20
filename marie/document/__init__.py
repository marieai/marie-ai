"""
Name : __init__.py

This import path is important to allow importing correctly as package
"""

from __future__ import absolute_import

import os
import sys

from .craft_ocr_processor import CraftOcrProcessor
from .trocr_ocr_processor import TrOcrProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
