"""CSV/TSV direct parse backend."""

import csv
import logging
from pathlib import Path

from marie.backend.base_backend import DocumentBackend

_log = logging.getLogger(__name__)

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LINE_HEIGHT = 14
LEFT_MARGIN = 36
COL_WIDTH = 80


class CsvBackend(DocumentBackend):
    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"csv", "tsv"}

    def convert(self, file_path: str, **kwargs) -> dict:
        path = Path(file_path)
        ext = path.suffix.lower()
        delimiter = "\t" if ext in (".tsv",) else ","

        text = path.read_text(encoding="utf-8")

        # Try to sniff the delimiter if csv
        if ext not in (".tsv",):
            try:
                dialect = csv.Sniffer().sniff(text[:4096], ",;\t|")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","

        reader = csv.reader(text.splitlines(), delimiter=delimiter)
        rows = list(reader)

        if not rows:
            return {"mode": "parsed", "results": [_empty_page(0)], "pages": 1}

        lines, all_words = _tabulate(rows)
        result = {
            "words": all_words,
            "lines": lines,
            "meta": {"page": 0, "width": PAGE_WIDTH, "height": PAGE_HEIGHT},
        }
        return {"mode": "parsed", "results": [result], "pages": 1}


def _tabulate(rows: list[list[str]]) -> tuple[list[dict], list[dict]]:
    """Convert rows into line/word structures with estimated bboxes."""
    y = LINE_HEIGHT
    all_lines = []
    all_words = []
    for row in rows:
        line_text = " | ".join(cell.strip() for cell in row)
        words = []
        x = LEFT_MARGIN
        for cell in row:
            cell_text = cell.strip()
            if not cell_text:
                x += COL_WIDTH
                continue
            for token in cell_text.split():
                w = len(token) * 7
                word = {
                    "text": token,
                    "bbox": [x, y, x + w, y + LINE_HEIGHT],
                    "confidence": 1.0,
                }
                words.append(word)
                x += w + 5
            x += 10  # gap between columns
        line_entry = {
            "text": line_text,
            "bbox": [LEFT_MARGIN, y, max(x, LEFT_MARGIN + 1), y + LINE_HEIGHT],
            "words": words,
        }
        all_lines.append(line_entry)
        all_words.extend(words)
        y += LINE_HEIGHT + 2
    return all_lines, all_words


def _empty_page(page_idx: int) -> dict:
    return {
        "words": [],
        "lines": [],
        "meta": {"page": page_idx, "width": PAGE_WIDTH, "height": PAGE_HEIGHT},
    }
