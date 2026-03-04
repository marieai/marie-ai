import logging
import os
import shutil
import subprocess
import tempfile

from marie.backend.base_backend import DocumentBackend

_log = logging.getLogger(__name__)


def _find_libreoffice() -> str | None:
    """Locate the LibreOffice/soffice binary."""
    cmd = shutil.which("libreoffice") or shutil.which("soffice")
    if cmd:
        return cmd
    # macOS app bundle fallback
    mac_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    if os.path.isfile(mac_path):
        return mac_path
    return None


class LibreOfficeBackend(DocumentBackend):
    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"doc", "xls", "ppt", "odt", "ods", "odp", "rtf"}

    @classmethod
    def is_available(cls) -> bool:
        return _find_libreoffice() is not None

    def convert(self, file_path: str, **kwargs) -> dict:
        lo_cmd = _find_libreoffice()
        if lo_cmd is None:
            raise RuntimeError("LibreOffice not found on this system")

        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    lo_cmd,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    tmpdir,
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
                raise RuntimeError(f"LibreOffice conversion did not produce {pdf_path}")

            from marie.backend.pdf_backend import PdfBackend

            return PdfBackend().convert(pdf_path, **kwargs)
