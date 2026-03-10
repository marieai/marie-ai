import logging

import numpy as np
from PIL import Image

from marie.backend.base_backend import DocumentBackend

_log = logging.getLogger(__name__)


class ImageBackend(DocumentBackend):
    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"png", "jpeg", "tiff", "bmp", "webp", "heif", "gif"}

    @classmethod
    def is_available(cls) -> bool:
        return True

    def convert(self, file_path: str, **kwargs) -> dict:
        img = Image.open(file_path)

        # Multi-frame images: TIFF (n_frames > 1) and animated GIF
        if getattr(img, "n_frames", 1) > 1:
            frames = []
            for i in range(img.n_frames):
                img.seek(i)
                frames.append(np.array(img.convert("RGB"), dtype=np.uint8))
            return {"mode": "frames", "frames": frames}

        return {
            "mode": "frames",
            "frames": [np.array(img.convert("RGB"), dtype=np.uint8)],
        }


def _register_heif():
    """Register HEIF/HEIC opener with Pillow. Call once at import time."""
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        _log.debug("pillow-heif not installed; HEIF/HEIC support unavailable")


_register_heif()
