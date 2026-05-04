from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BatchResult:
    task_id: str
    response: Optional[str]
    error: Optional[Exception]
