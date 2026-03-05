"""HTML/XML direct parse backend."""

import logging
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from marie.backend.base_backend import DocumentBackend

_log = logging.getLogger(__name__)

_TEXT_TAGS = {
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "td",
    "th",
    "dt",
    "dd",
    "pre",
    "blockquote",
    "figcaption",
    "caption",
    "summary",
    "address",
}

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LINE_HEIGHT = 14
LEFT_MARGIN = 36


class HtmlBackend(DocumentBackend):
    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"html", "xml"}

    def convert(self, file_path: str, **kwargs) -> dict:
        path = Path(file_path)
        raw = path.read_bytes()
        parser = "lxml-xml" if path.suffix.lower() == ".xml" else "html.parser"
        soup = BeautifulSoup(raw, parser)

        # Remove non-content elements
        for tag in soup(["script", "noscript", "style"]):
            tag.decompose()

        lines = _extract_lines(soup)
        results = [_build_page(lines, 0)]
        return {"mode": "parsed", "results": results, "pages": 1}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import bs4  # noqa: F401

            return True
        except ImportError:
            return False


def _extract_lines(soup: BeautifulSoup) -> list[str]:
    """Extract text lines from block-level elements."""
    lines: list[str] = []
    content = soup.body or soup
    for el in content.find_all(_TEXT_TAGS):
        if not isinstance(el, Tag):
            continue
        text = el.get_text(separator=" ", strip=True)
        if text:
            lines.append(text)
    return lines


def _build_page(lines: list[str], page_idx: int) -> dict:
    """Build a page result dict with estimated bounding boxes."""
    y = LINE_HEIGHT
    page_lines = []
    all_words = []
    for line_text in lines:
        words = []
        x = LEFT_MARGIN
        for token in line_text.split():
            w = len(token) * 7  # rough char width estimate
            word = {
                "text": token,
                "bbox": [x, y, x + w, y + LINE_HEIGHT],
                "confidence": 1.0,
            }
            words.append(word)
            x += w + 5
        line_entry = {
            "text": line_text,
            "bbox": [LEFT_MARGIN, y, max(x, LEFT_MARGIN + 1), y + LINE_HEIGHT],
            "words": words,
        }
        page_lines.append(line_entry)
        all_words.extend(words)
        y += LINE_HEIGHT + 2
    return {
        "words": all_words,
        "lines": page_lines,
        "meta": {"page": page_idx, "width": PAGE_WIDTH, "height": PAGE_HEIGHT},
    }
