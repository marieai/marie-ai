"""MarkItDown semantic extraction provider."""

from __future__ import annotations

from importlib import metadata
from importlib.util import find_spec
from io import BytesIO

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
        'pdf': ('pdfminer', 'pypdfium2'),
        'docx': ('docx',),
        'pptx': ('pptx',),
        'xlsx': ('openpyxl',),
    }

    def is_ready(self, canonical_format: str) -> bool:
        if canonical_format not in self.formats or find_spec('markitdown') is None:
            return False
        dependencies = self._readiness_imports.get(canonical_format, ())
        return all(find_spec(dependency) is not None for dependency in dependencies)

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
        if canonical_format == 'pdf':
            result, content, page_count = _convert_pdf(path)
        else:
            from markitdown import MarkItDown

            result = MarkItDown().convert(path)
            content = _result_text(result)
            page_count = None
        if not content.strip():
            raise ProviderNotExtractableError(
                f'MarkItDown returned no content for {canonical_format}'
            )
        title = getattr(result, 'title', None)
        metadata_value = {'title': title.strip()} if isinstance(title, str) else {}
        if page_count is not None:
            metadata_value['page_count'] = page_count
        return ProviderDocument(
            content=content,
            provider=self.provider_id,
            provider_version=metadata.version('markitdown'),
            backend=type(result).__name__,
            metadata=metadata_value,
        )


def _convert_pdf(path: str) -> tuple[object, str, int]:
    from markitdown import MarkItDown, StreamInfo
    from pypdfium2 import PdfDocument, PdfiumError

    converter = MarkItDown()
    result = converter.convert(path)
    content = _result_text(result)

    try:
        source = PdfDocument(path)
        try:
            page_count = len(source)
            if _content_page_count(content) == page_count:
                return result, content, page_count

            pages = []
            for page_index in range(page_count):
                output = BytesIO()
                page_document = PdfDocument.new()
                try:
                    page_document.import_pages(source, [page_index])
                    page_document.save(output)
                finally:
                    page_document.close()
                output.seek(0)
                page_result = converter.convert(
                    output,
                    stream_info=StreamInfo(extension='.pdf'),
                )
                pages.append(_result_text(page_result).strip('\f'))
        finally:
            source.close()
    except (OSError, PdfiumError) as error:
        raise ProviderNotExtractableError(
            'MarkItDown could not preserve PDF page boundaries'
        ) from error

    return result, '\f'.join(pages), page_count


def _content_page_count(content: str) -> int:
    pages = content.split('\f')
    if pages and not pages[-1].strip():
        pages.pop()
    return len(pages) or 1


def _result_text(result: object) -> str:
    for attribute in ('text_content', 'markdown'):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            return value
    return str(result)
