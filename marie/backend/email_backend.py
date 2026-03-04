"""Email (EML/MSG) direct parse backend."""

import email
import logging
from email import policy
from pathlib import Path

from marie.backend.base_backend import DocumentBackend

_log = logging.getLogger(__name__)

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LINE_HEIGHT = 14
LEFT_MARGIN = 36


class EmailBackend(DocumentBackend):
    @classmethod
    def supported_formats(cls) -> set[str]:
        return {"eml", "msg"}

    def convert(self, file_path: str, **kwargs) -> dict:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".msg":
            lines = _parse_msg(path)
        else:
            lines = _parse_eml(path)

        results = [_build_page(lines, 0)]
        return {"mode": "parsed", "results": results, "pages": 1}

    @classmethod
    def is_available(cls) -> bool:
        # EML uses stdlib; MSG needs extract_msg
        try:
            import extract_msg  # noqa: F401

            return True
        except ImportError:
            return False


def _parse_eml(path: Path) -> list[str]:
    """Parse an EML file using the stdlib email module."""
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    lines: list[str] = []
    # Headers
    for hdr in ("Subject", "From", "To", "Date"):
        val = msg.get(hdr)
        if val:
            lines.append(f"{hdr}: {val}")

    # Body
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is not None:
        content = body.get_content()
        if isinstance(content, str):
            for line in content.splitlines():
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)

    # Attachment names
    for part in msg.iter_attachments():
        fname = part.get_filename()
        if fname:
            lines.append(f"[Attachment: {fname}]")

    return lines


def _parse_msg(path: Path) -> list[str]:
    """Parse an MSG file using extract-msg."""
    import extract_msg

    msg = extract_msg.Message(str(path))
    lines: list[str] = []
    if msg.subject:
        lines.append(f"Subject: {msg.subject}")
    if msg.sender:
        lines.append(f"From: {msg.sender}")
    if msg.to:
        lines.append(f"To: {msg.to}")
    if msg.date:
        lines.append(f"Date: {msg.date}")

    if msg.body:
        for line in msg.body.splitlines():
            stripped = line.strip()
            if stripped:
                lines.append(stripped)

    for att in msg.attachments:
        if hasattr(att, "longFilename") and att.longFilename:
            lines.append(f"[Attachment: {att.longFilename}]")
    msg.close()
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
