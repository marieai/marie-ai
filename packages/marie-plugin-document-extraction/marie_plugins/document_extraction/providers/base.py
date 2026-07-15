"""Provider adapter contract."""

from __future__ import annotations

from typing import Protocol

from ..models import ProviderDocument


class ProviderUnavailableError(RuntimeError):
    """Raised when an optional provider edge is not installed or ready."""


class ProviderNotExtractableError(RuntimeError):
    """Raised when a provider cannot produce useful semantic content."""


class ExtractionProvider(Protocol):
    provider_id: str
    formats: frozenset[str]
    output_formats: frozenset[str]

    def is_ready(self, canonical_format: str) -> bool: ...

    def extract(
        self,
        path: str,
        canonical_format: str,
        options: dict | None = None,
        output_format: str = 'markdown',
    ) -> ProviderDocument: ...
