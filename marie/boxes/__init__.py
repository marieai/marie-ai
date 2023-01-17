"""
Name : __init__.py

This import path is important to allow importing correctly as package
"""

from __future__ import absolute_import

import os
import sys

from .box_processor import PSMode
from .craft_box_processor import BoxProcessorCraft
from .dit.ulim_dit_box_processor import BoxProcessorUlimDit

# from .textfusenet_box_processor import BoxProcessorTextFuseNet

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
