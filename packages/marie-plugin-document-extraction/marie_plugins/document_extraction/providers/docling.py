"""Docling Slim semantic extraction provider."""

from __future__ import annotations

import json
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from ..models import ProviderDocument, ResultKind
from .base import (
    ProviderNotExtractableError,
    ProviderUnavailableError,
)

_EXPORTS = {
    'markdown': ('export_to_markdown', 'text/markdown', ResultKind.SEMANTIC_DOCUMENT),
    'html': ('export_to_html', 'text/html', ResultKind.SEMANTIC_DOCUMENT),
    'text': ('export_to_text', 'text/plain', ResultKind.SEMANTIC_DOCUMENT),
    'json': ('export_to_dict', 'application/json', ResultKind.STRUCTURED_DOCUMENT),
}


class DoclingProvider:
    provider_id = 'docling'
    output_formats = frozenset(_EXPORTS)
    formats = frozenset(
        {
            'docx',
            'pptx',
            'xlsx',
            'odt',
            'ods',
            'odp',
            'html',
            'markdown',
            'csv',
            'latex',
            'eml',
        }
    )

    _readiness_imports = {
        'docx': 'docx',
        'pptx': 'pptx',
        'xlsx': 'openpyxl',
        'odt': 'odfdo',
        'ods': 'odfdo',
        'odp': 'odfdo',
        'html': 'bs4',
        'markdown': 'marko',
        'latex': 'pylatexenc',
        'eml': 'mailparser',
    }

    def is_ready(self, canonical_format: str) -> bool:
        if canonical_format not in self.formats or find_spec('docling') is None:
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
                f'Docling is not ready for {canonical_format}'
            )
        if output_format not in _EXPORTS:
            raise ValueError(f'Docling cannot produce {output_format!r} output')

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import InputDocument

        input_format = _input_format(InputFormat, canonical_format)
        backend_type = _backend_type(canonical_format)
        input_document = InputDocument(
            path_or_stream=Path(path),
            format=input_format,
            backend=backend_type,
        )
        backend = input_document._backend
        if not input_document.valid or not backend.is_valid():
            raise ProviderNotExtractableError(
                f'Docling rejected the {canonical_format} input'
            )
        exporter, media_type, result_kind = _EXPORTS[output_format]
        document = backend.convert()
        exported = getattr(document, exporter)()
        content = (
            json.dumps(exported, ensure_ascii=False)
            if output_format == 'json'
            else exported
        )
        if not content.strip():
            raise ProviderNotExtractableError(
                f'Docling returned no content for {canonical_format}'
            )
        return ProviderDocument(
            content=content,
            media_type=media_type,
            result_kind=result_kind,
            provider=self.provider_id,
            provider_version=metadata.version('docling-slim'),
            backend=backend_type.__name__,
            metadata={'status': 'success'},
        )


def _input_format(input_format, canonical_format: str):
    mapping = {
        'docx': input_format.DOCX,
        'pptx': input_format.PPTX,
        'xlsx': input_format.XLSX,
        'odt': input_format.ODT,
        'ods': input_format.ODS,
        'odp': input_format.ODP,
        'html': input_format.HTML,
        'markdown': input_format.MD,
        'csv': input_format.CSV,
        'latex': input_format.LATEX,
        'eml': input_format.EMAIL,
    }
    return mapping[canonical_format]


def _backend_type(canonical_format: str):
    from docling.backend.csv_backend import CsvDocumentBackend
    from docling.backend.email_backend import EmailDocumentBackend
    from docling.backend.html_backend import HTMLDocumentBackend
    from docling.backend.latex_backend import LatexDocumentBackend
    from docling.backend.md_backend import MarkdownDocumentBackend
    from docling.backend.msexcel_backend import MsExcelDocumentBackend
    from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
    from docling.backend.msword_backend import MsWordDocumentBackend
    from docling.backend.opendocument_backend import (
        OdpDocumentBackend,
        OdsDocumentBackend,
        OdtDocumentBackend,
    )

    mapping = {
        'csv': CsvDocumentBackend,
        'docx': MsWordDocumentBackend,
        'eml': EmailDocumentBackend,
        'html': HTMLDocumentBackend,
        'latex': LatexDocumentBackend,
        'markdown': MarkdownDocumentBackend,
        'odp': OdpDocumentBackend,
        'ods': OdsDocumentBackend,
        'odt': OdtDocumentBackend,
        'pptx': MsPowerpointDocumentBackend,
        'xlsx': MsExcelDocumentBackend,
    }
    return mapping[canonical_format]
