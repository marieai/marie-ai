"""EPUB direct parse backend."""

import logging
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, Tag
from ebooklib import epub

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
    "pre",
    "blockquote",
}

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LINE_HEIGHT = 14
LEFT_MARGIN = 36


class EpubBackend(DocumentBackend):
    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"epub"}

    def convert(self, file_path: str, **kwargs) -> dict:
        book = epub.read_epub(file_path, options={"ignore_ncx": True})
        results = []
        page_idx = 0

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html_content = item.get_content()
            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(["script", "noscript", "style"]):
                tag.decompose()

            lines = _extract_lines(soup)
            if not lines:
                continue
            results.append(_build_page(lines, page_idx))
            page_idx += 1

        if not results:
            results.append(_empty_page(0))
            page_idx = 1

        return {"mode": "parsed", "results": results, "pages": page_idx}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import ebooklib  # noqa: F401

            return True
        except ImportError:
            return False


def _extract_lines(soup: BeautifulSoup) -> list[str]:
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
    y = LINE_HEIGHT
    page_lines = []
    all_words = []
    for line_text in lines:
        words = []
        x = LEFT_MARGIN
        for token in line_text.split():
            w = len(token) * 7
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


def _empty_page(page_idx: int) -> dict:
    return {
        "words": [],
        "lines": [],
        "meta": {"page": page_idx, "width": PAGE_WIDTH, "height": PAGE_HEIGHT},
    }
