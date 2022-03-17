"""
Name : __init__.py

This import path is important to allow importing correctly as package
"""

from __future__ import absolute_import

import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
