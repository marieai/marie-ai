from __future__ import absolute_import

import os
import sys

from .renderer import ResultRenderer

from .text_renderer import TextRenderer  # isort:skip depends on ResultRenderer
from .pdf_renderer import PdfRenderer  # isort:skip depends on ResultRenderer
from .tiff_renderer import TiffRenderer  # isort:skip depends on ResultRenderer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
