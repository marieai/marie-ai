"""Markdown direct parse backend."""

import logging
from pathlib import Path

import marko
import marko.block
import marko.inline

from marie.backend.base_backend import DocumentBackend

_log = logging.getLogger(__name__)

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LINE_HEIGHT = 14
LEFT_MARGIN = 36


class MdBackend(DocumentBackend):
    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"markdown"}

    def convert(self, file_path: str, **kwargs) -> dict:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        parser = marko.Markdown()
        ast = parser.parse(text)
        lines = _extract_lines(ast)
        results = [_build_page(lines, 0)]
        return {"mode": "parsed", "results": results, "pages": 1}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import marko  # noqa: F401

            return True
        except ImportError:
            return False


def _extract_lines(element) -> list[str]:
    """Recursively extract text lines from the marko AST."""
    lines: list[str] = []

    if isinstance(element, (marko.block.Heading, marko.block.SetextHeading)):
        text = _inline_text(element)
        if text:
            lines.append(text)
        return lines

    if isinstance(element, marko.block.Paragraph):
        text = _inline_text(element)
        if text:
            lines.append(text)
        return lines

    if isinstance(element, (marko.block.CodeBlock, marko.block.FencedCode)):
        if element.children and isinstance(element.children[0], marko.inline.RawText):
            raw = element.children[0].children
            if isinstance(raw, str):
                for line in raw.strip().splitlines():
                    stripped = line.strip()
                    if stripped:
                        lines.append(stripped)
        return lines

    if hasattr(element, "children") and not isinstance(element.children, str):
        for child in element.children:
            lines.extend(_extract_lines(child))

    return lines


def _inline_text(element) -> str:
    """Flatten inline children to plain text."""
    parts: list[str] = []
    if not hasattr(element, "children"):
        return ""
    for child in element.children:
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, marko.inline.RawText):
            if isinstance(child.children, str):
                parts.append(child.children)
        elif isinstance(child, marko.inline.CodeSpan):
            if isinstance(child.children, str):
                parts.append(child.children)
        elif hasattr(child, "children"):
            parts.append(_inline_text(child))
    return " ".join("".join(parts).split())


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
