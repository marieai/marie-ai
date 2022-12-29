"""
This is the current version
"""

import os
import pkgutil
from pathlib import Path

# version_path = os.path.join(Path(__file__).resolve().parent.parent, 'version.txt')
# current_pkg_version = pkgutil.get_data(__name__, "version.txt").decode("ascii").strip()
# __version__ = current_pkg_version
# __version__ = open(version_path, 'r').read().strip()

__version__ = '3.0.0'
