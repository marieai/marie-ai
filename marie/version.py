"""
This is the current version
"""

import os
from pathlib import Path

version_path = os.path.join(Path(__file__).resolve().parent.parent, 'version.txt')
# print(f'version_path : {version_path}')
__version__ = open(version_path, 'r').read().strip()
