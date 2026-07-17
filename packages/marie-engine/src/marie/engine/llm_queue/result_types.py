from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BatchResult:
    task_id: str
    response: Optional[Any]
    error: Optional[Exception]
