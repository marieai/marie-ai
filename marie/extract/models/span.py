from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Span:
    """
    Pure span for a single page using line-index coordinates.

    y:   start line id (inclusive)
    h:   height in lines (end = y + h, exclusive)

    NOTE: We keep (y, h) for backward-compatibility with existing code.
          For readability you can also use the properties below.
    """

    page: int = 0
    y: int = 0
    h: int = 0
    msg: Optional[str] = None

    @property
    def start_line_id(self) -> int:
        return self.y

    @property
    def end_line_id(self) -> int:
        return self.y + self.h  # exclusive

    def with_msg(self, message: Optional[str]) -> "Span":
        self.msg = message
        return self
