import os
import shutil
from datetime import datetime
from typing import List

from marie.common.file_io import get_cache_dir
from marie.utils.image_utils import hash_frames_fast
from marie.utils.utils import ensure_exists


def create_working_dir(frames: List, backup: bool = False) -> str:
    frame_checksum = hash_frames_fast(frames=frames)
    generators_dir = os.path.join(get_cache_dir(), "generators")
    os.makedirs(generators_dir, exist_ok=True)

    # create backup name by appending a timestamp
    if backup:
        if os.path.exists(os.path.join(generators_dir, frame_checksum)):
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            shutil.move(
                os.path.join(generators_dir, frame_checksum),
                os.path.join(generators_dir, f"{frame_checksum}-{ts}"),
            )
    return ensure_exists(os.path.join(generators_dir, frame_checksum))
