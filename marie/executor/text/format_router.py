"""Pure routing helpers deciding whether a source extracts via markitdown or OCR.

``route`` maps a detected document format to an engine; ``pdf_yield_ok`` gates a
markitdown PDF result against a per-page character floor so scanned PDFs (which
convert to near-empty markdown) fall through to OCR.
"""

from __future__ import annotations

# Digital formats markitdown extracts directly (canonical names from
# ``marie.utils.format_registry`` plus the raw spellings the plan lists). Legacy
# office (doc/xls/ppt), ods/odp, xml/tsv, and render-only formats intentionally
# stay on the OCR/rasterize path.
_DIGITAL_FORMATS = frozenset(
    {
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "html",
        "htm",
        "md",
        "markdown",
        "csv",
        "rtf",
        "odt",
        "epub",
        "eml",
        "msg",
    }
)


class FormatRouter:
    """Stateless format→engine router."""

    @staticmethod
    def route(file_type: str, parse_mode: str | None) -> str:
        """Return ``"markitdown"`` for born-digital formats, else ``"ocr"``.

        ``parse_mode == "ocr"`` forces OCR for any format. Images and any format
        not in the digital set route to OCR.
        """
        if parse_mode == "ocr":
            return "ocr"
        if (file_type or "").lower() in _DIGITAL_FORMATS:
            return "markitdown"
        return "ocr"


def pdf_yield_ok(
    markdown: str, page_count: int, floor_chars_per_page: int = 200
) -> bool:
    """Whether a markitdown PDF result clears the per-page text-yield floor.

    A PDF with no text layer converts to near-empty markdown; below the floor it
    should fall through to OCR. ``page_count`` is clamped to at least 1.
    """
    pages = page_count if page_count and page_count > 0 else 1
    return len(markdown or "") >= floor_chars_per_page * pages
