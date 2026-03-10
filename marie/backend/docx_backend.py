import logging

from docx import Document
from docx.table import Table

from marie.backend.base_backend import DocumentBackend

_log = logging.getLogger(__name__)

# Default A4 page dimensions in points
_DEFAULT_WIDTH = 612.0
_DEFAULT_HEIGHT = 792.0

# EMU to points conversion factor
_EMU_TO_PT = 1.0 / 12700.0


def _emu_to_pt(emu) -> float:
    if emu is None:
        return 0.0
    return float(emu) * _EMU_TO_PT


def _text_to_words(text: str, x0: float, y0: float, x1: float, y1: float) -> list[dict]:
    """Split text into word dicts with estimated bounding boxes spread across the line."""
    parts = text.split()
    if not parts:
        return []
    total_chars = sum(len(w) for w in parts)
    if total_chars == 0:
        return []
    span = x1 - x0
    words = []
    cx = x0
    for w in parts:
        frac = len(w) / total_chars
        wx1 = cx + span * frac
        words.append(
            {
                "text": w,
                "bbox": [round(cx, 2), round(y0, 2), round(wx1, 2), round(y1, 2)],
                "confidence": 1.0,
            }
        )
        cx = wx1
    return words


class DocxBackend(DocumentBackend):
    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"docx"}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import docx  # noqa: F401

            return True
        except ImportError:
            return False

    def convert(self, file_path: str, **kwargs) -> dict:
        doc = Document(file_path)

        # Page dimensions from first section, fall back to A4
        section = doc.sections[0] if doc.sections else None
        page_w = _emu_to_pt(getattr(section, "page_width", None)) or _DEFAULT_WIDTH
        page_h = _emu_to_pt(getattr(section, "page_height", None)) or _DEFAULT_HEIGHT

        lines: list[dict] = []
        y_cursor = 36.0  # start below top margin
        line_height = 14.0
        margin_left = 36.0
        text_width = page_w - 2 * margin_left

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "tbl":
                table = Table(element, doc)
                for row in table.rows:
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if not cell_text:
                            continue
                        x0, x1 = margin_left, margin_left + text_width
                        y0, y1 = y_cursor, y_cursor + line_height
                        words = _text_to_words(cell_text, x0, y0, x1, y1)
                        if words:
                            lines.append(
                                {
                                    "text": cell_text,
                                    "bbox": [
                                        round(x0, 2),
                                        round(y0, 2),
                                        round(x1, 2),
                                        round(y1, 2),
                                    ],
                                    "words": words,
                                }
                            )
                        y_cursor += line_height

            elif tag == "p":
                from docx.text.paragraph import Paragraph

                para = Paragraph(element, doc)
                text = para.text.strip()
                if not text:
                    y_cursor += line_height * 0.5
                    continue
                x0, x1 = margin_left, margin_left + text_width
                y0, y1 = y_cursor, y_cursor + line_height
                words = _text_to_words(text, x0, y0, x1, y1)
                if words:
                    lines.append(
                        {
                            "text": text,
                            "bbox": [
                                round(x0, 2),
                                round(y0, 2),
                                round(x1, 2),
                                round(y1, 2),
                            ],
                            "words": words,
                        }
                    )
                y_cursor += line_height

        result = {
            "words": [w for line in lines for w in line["words"]],
            "lines": lines,
            "meta": {"page": 0, "width": page_w, "height": page_h},
        }
        return {"mode": "parsed", "results": [result], "pages": 1}
