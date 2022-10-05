"""
Name : __init__.py

This import path is important to allow importing correctly as package
"""

from __future__ import absolute_import

import os
import sys

from .craft_box_processor import BoxProcessorCraft
from .textfusenet_box_processor import BoxProcessorTextFuseNet
from .dit.ulim_dit_box_processor import BoxProcessorUlimDit

from .box_processor import PSMode

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
