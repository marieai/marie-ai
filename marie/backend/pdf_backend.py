import logging
import os
import tempfile

import cv2
import numpy as np
from PIL import Image

from marie.backend.base_backend import DocumentBackend

_log = logging.getLogger(__name__)


class PdfBackend(DocumentBackend):
    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"pdf"}

    def convert(self, file_path: str, **kwargs) -> dict:
        frames = self._extract_xobjects(file_path)
        if frames and not self._all_blank(frames):
            return {"mode": "frames", "frames": frames}
        _log.debug("XObject extraction empty or blank; falling back to page rendering")
        return {"mode": "frames", "frames": self._render_pages(file_path, **kwargs)}

    def _extract_xobjects(self, file_path: str) -> list[np.ndarray]:
        """Extract embedded images from PDF XObjects (ported from marie/utils/docs.py)."""
        try:
            import PyPDF4
            from PyPDF4 import PdfFileReader
        except ImportError:
            _log.debug("PyPDF4 not available; skipping XObject extraction")
            return []

        frames = []
        try:
            with open(file_path, "rb") as f:
                pdf = PdfFileReader(f)
                for page_index in range(pdf.getNumPages()):
                    page = pdf.getPage(page_index)
                    size = (
                        int(page.mediaBox.getWidth()),
                        int(page.mediaBox.getHeight()),
                    )
                    resources = page["/Resources"]

                    if "/XObject" not in resources and "/ProcSet" not in resources:
                        continue

                    x_object = (
                        resources.get("/XObject", {}).getObject()
                        if "/XObject" in resources
                        else {}
                    )

                    for obj in x_object:
                        if x_object[obj]["/Subtype"] == "/Image":
                            size = (x_object[obj]["/Width"], x_object[obj]["/Height"])
                            data = x_object[obj].getData()
                            mode = (
                                "RGB"
                                if x_object[obj]["/ColorSpace"] == "/DeviceRGB"
                                else "P"
                            )
                            img = self._decode_xobject(data, mode, size)
                            if img is not None:
                                frames.append(img)
                        else:
                            blank = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
                            frames.append(blank)
        except Exception:
            _log.debug("XObject extraction failed", exc_info=True)
            return []
        return frames

    @staticmethod
    def _decode_xobject(data: bytes, mode: str, size: tuple) -> np.ndarray | None:
        """Decode raw XObject image data into a numpy array."""
        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(data)

            # Try TIFF first (CCITT encoded)
            loaded, tiff_frames = cv2.imreadmulti(path, [], cv2.IMREAD_ANYCOLOR)
            if loaded and tiff_frames:
                img = tiff_frames[0]
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                return img

            # Generic image read
            img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
            return img
        except Exception:
            try:
                img = Image.frombytes(mode, size, data)
                return np.array(img.convert("RGB"), dtype=np.uint8)
            except Exception:
                _log.debug("Failed to decode XObject image data", exc_info=True)
                return None
        finally:
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _all_blank(frames: list[np.ndarray], threshold: float = 0.99) -> bool:
        """Check if all frames are blank (nearly all white)."""
        for frame in frames:
            white_ratio = np.mean(frame > 250)
            if white_ratio < threshold:
                return False
        return True

    @staticmethod
    def _render_pages(file_path: str, dpi: int = 200, **kwargs) -> list[np.ndarray]:
        """Render PDF pages to images using poppler via pdf2image."""
        from pdf2image import convert_from_path

        images = convert_from_path(file_path, dpi=dpi)
        return [np.array(img.convert("RGB"), dtype=np.uint8) for img in images]
