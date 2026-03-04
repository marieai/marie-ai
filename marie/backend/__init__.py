from marie.backend.base_backend import DocumentBackend
from marie.backend.csv_backend import CsvBackend
from marie.backend.docx_backend import DocxBackend
from marie.backend.email_backend import EmailBackend
from marie.backend.epub_backend import EpubBackend
from marie.backend.html_backend import HtmlBackend
from marie.backend.image_backend import ImageBackend
from marie.backend.libreoffice_backend import LibreOfficeBackend
from marie.backend.md_backend import MdBackend
from marie.backend.pdf_backend import PdfBackend
from marie.backend.pptx_backend import PptxBackend
from marie.backend.special_backend import DjvuBackend, LatexBackend, RstBackend
from marie.backend.xlsx_backend import XlsxBackend

_BACKENDS: dict[str, type[DocumentBackend]] = {}
for _cls in [
    PdfBackend,
    ImageBackend,
    DocxBackend,
    PptxBackend,
    XlsxBackend,
    HtmlBackend,
    MdBackend,
    CsvBackend,
    EmailBackend,
    EpubBackend,
    LibreOfficeBackend,
    RstBackend,
    LatexBackend,
    DjvuBackend,
]:
    for _fmt in _cls.supported_formats():
        _BACKENDS[_fmt] = _cls


def get_backend(format_name: str) -> DocumentBackend:
    cls = _BACKENDS.get(format_name)
    if cls is None:
        raise ValueError(f"No backend for format: {format_name}")
    if not cls.is_available():
        raise ImportError(
            f"Backend for '{format_name}' is not available (missing system dependency)"
        )
    return cls()
