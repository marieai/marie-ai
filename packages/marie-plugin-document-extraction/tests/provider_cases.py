"""Required real-provider checks for the plugin uv environment."""

from pathlib import Path

import pytest
from marie_plugins.document_extraction.providers.docling import DoclingProvider
from marie_plugins.document_extraction.providers.markitdown import MarkItDownProvider

FIXTURES = Path(__file__).parent / 'fixtures'

DOCLING_CASES = {
    'csv': ('sample.csv', 'Accounting/Finance'),
    'docx': ('sample.docx', 'Some text'),
    'eml': ('sample.eml', 'This is a simple email body.'),
    'html': ('sample.html', 'This is some text.'),
    'latex': ('sample.tex', 'Sample Document'),
    'markdown': ('sample.md', 'Some heading'),
    'odp': ('sample.odp', 'Test Table Slide'),
    'ods': ('sample.ods', 'Freshwater Ducks'),
    'odt': ('sample.odt', 'Lorem Ipsum'),
    'pptx': ('sample.pptx', 'Test Table Slide'),
    'xlsx': ('sample.xlsx', 'data3'),
}

MARKITDOWN_CASES = {
    'csv': ('sample.csv', 'Accounting/Finance'),
    'docx': ('sample.docx', 'Some text'),
    'epub': ('sample.epub', 'Sarah Louisa Forten Purvis'),
    'html': ('sample.html', 'This is some text.'),
    'pdf': ('sample.pdf', 'PART I - DEFINITIONS'),
    'pptx': ('sample.pptx', 'Test Table Slide'),
    'xlsx': ('sample.xlsx', 'data3'),
}


@pytest.mark.parametrize(
    ('canonical_format', 'fixture_and_token'), DOCLING_CASES.items()
)
def test_docling_provider_edges(
    canonical_format: str, fixture_and_token: tuple[str, str]
) -> None:
    provider = DoclingProvider()
    assert provider.formats == frozenset(DOCLING_CASES)
    assert provider.is_ready(canonical_format)

    fixture, expected_token = fixture_and_token
    result = provider.extract(str(FIXTURES / fixture), canonical_format)

    assert result.provider == 'docling'
    assert result.provider_version == '2.111.0'
    assert expected_token in result.content


@pytest.mark.parametrize(
    ('canonical_format', 'fixture_and_token'), MARKITDOWN_CASES.items()
)
def test_markitdown_provider_edges(
    canonical_format: str, fixture_and_token: tuple[str, str]
) -> None:
    provider = MarkItDownProvider()
    assert provider.formats == frozenset(MARKITDOWN_CASES)
    assert provider.is_ready(canonical_format)

    fixture, expected_token = fixture_and_token
    result = provider.extract(str(FIXTURES / fixture), canonical_format)

    assert result.provider == 'markitdown'
    assert result.provider_version == '0.1.6'
    assert expected_token in result.content
