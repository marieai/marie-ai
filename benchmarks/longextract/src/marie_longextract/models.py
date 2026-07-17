from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PartialExtraction:
    unit_name: str
    page_index: int
    rows: list[dict[str, Any]]
    scalars: dict[str, Any]
    source_uri: str


@dataclass(frozen=True)
class VerificationFinding:
    code: str
    unit_name: str
    page_index: int | None
    message: str
    repairable: bool
