"""MarkItDown semantic extraction provider."""

from __future__ import annotations

from importlib import metadata
from importlib.util import find_spec

from ..models import ProviderDocument
from .base import (
    ProviderNotExtractableError,
    ProviderUnavailableError,
)


class MarkItDownProvider:
    provider_id = 'markitdown'
    formats = frozenset({'pdf', 'docx', 'pptx', 'xlsx', 'html', 'csv', 'epub'})
    output_formats = frozenset({'markdown'})

    _readiness_imports = {
        'pdf': 'pdfminer',
        'docx': 'docx',
        'pptx': 'pptx',
        'xlsx': 'openpyxl',
    }

    def is_ready(self, canonical_format: str) -> bool:
        if canonical_format not in self.formats or find_spec('markitdown') is None:
            return False
        dependency = self._readiness_imports.get(canonical_format)
        return dependency is None or find_spec(dependency) is not None

    def extract(
        self,
        path: str,
        canonical_format: str,
        options: dict | None = None,
        output_format: str = 'markdown',
    ) -> ProviderDocument:
        if not self.is_ready(canonical_format):
            raise ProviderUnavailableError(
                f'MarkItDown is not ready for {canonical_format}'
            )
        if output_format not in self.output_formats:
            raise ValueError(f'MarkItDown cannot produce {output_format!r} output')
        from markitdown import MarkItDown

        result = MarkItDown().convert(path)
        content = _result_text(result)
        if not content.strip():
            raise ProviderNotExtractableError(
                f'MarkItDown returned no content for {canonical_format}'
            )
        title = getattr(result, 'title', None)
        metadata_value = {'title': title.strip()} if isinstance(title, str) else {}
        return ProviderDocument(
            content=content,
            provider=self.provider_id,
            provider_version=metadata.version('markitdown'),
            backend=type(result).__name__,
            metadata=metadata_value,
        )


def _result_text(result: object) -> str:
    for attribute in ('text_content', 'markdown'):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            return value
    return str(result)
