import logging
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from marie.backend.base_backend import DocumentBackend

_log = logging.getLogger(__name__)


class RstBackend(DocumentBackend):
    """ReStructuredText: docutils -> HTML -> weasyprint -> PDF -> frames."""

    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"rst"}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import docutils  # noqa: F401
            import weasyprint  # noqa: F401

            return True
        except ImportError:
            return False

    def convert(self, file_path: str, **kwargs) -> dict:
        from docutils.core import publish_string
        from weasyprint import HTML

        with open(file_path, "r", encoding="utf-8") as f:
            rst_source = f.read()

        html_bytes = publish_string(rst_source, writer_name="html")
        pdf_bytes = HTML(string=html_bytes.decode("utf-8")).write_pdf()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            from pdf2image import convert_from_path

            images = convert_from_path(tmp_path, dpi=200)
            frames = [np.array(img.convert("RGB"), dtype=np.uint8) for img in images]
        finally:
            os.unlink(tmp_path)

        return {"mode": "frames", "frames": frames}


class LatexBackend(DocumentBackend):
    """LaTeX: pdflatex -> PDF -> frames."""

    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"latex"}

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("pdflatex") is not None

    def convert(self, file_path: str, **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    f"-output-directory={tmpdir}",
                    file_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=120,
            )

            base = os.path.splitext(os.path.basename(file_path))[0]
            pdf_path = os.path.join(tmpdir, base + ".pdf")
            if not os.path.exists(pdf_path):
                raise RuntimeError(f"pdflatex did not produce {pdf_path}")

            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, dpi=200)
            return {
                "mode": "frames",
                "frames": [
                    np.array(img.convert("RGB"), dtype=np.uint8) for img in images
                ],
            }


class DjvuBackend(DocumentBackend):
    """DjVu: ddjvu CLI -> TIFF -> OpenCV frames."""

    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"djvu"}

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("ddjvu") is not None

    def convert(self, file_path: str, **kwargs) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
            tiff_path = tmp.name

        try:
            subprocess.run(
                ["ddjvu", "-format=tiff", file_path, tiff_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=120,
            )

            loaded, tiff_frames = cv2.imreadmulti(tiff_path, [], cv2.IMREAD_ANYCOLOR)
            if not loaded:
                raise RuntimeError(
                    f"Failed to read TIFF output from ddjvu: {tiff_path}"
                )

            frames = []
            for frame in tiff_frames:
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            if os.path.exists(tiff_path):
                os.unlink(tiff_path)

        return {"mode": "frames", "frames": frames}
