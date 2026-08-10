from marie.extract.results.core.core_parsers import (
    _adapt_tables_to_table_extraction_result,
)
from marie.extract.structures import UnstructuredDocument


def test_adapt_tables_accepts_single_extraction_object() -> None:
    document = UnstructuredDocument(
        lines=[],
        regions=None,
        metadata={'source_metadata': {'pages': 1}},
    )
    table = {
        'name': 'Table 1',
        'header_rows': [],
        'rows': [],
        'columns': ['Service date', 'Amount'],
    }

    result = _adapt_tables_to_table_extraction_result(
        {'extractions': table}, document, page_id=0
    )

    assert result['extractions'] == [
        {
            'name': 'Table 1',
            'header_rows': [],
            'rows': [],
            'columns': ['Service date', 'Amount'],
            'table_classification': None,
            'page_index': None,
            'header_present': None,
            'continuation': None,
            'columns_inferred': None,
        }
    ]


def test_adapt_tables_skips_non_object_extraction() -> None:
    document = UnstructuredDocument(
        lines=[],
        regions=None,
        metadata={'source_metadata': {'pages': 1}},
    )

    result = _adapt_tables_to_table_extraction_result(
        {'extractions': ['not a table']}, document, page_id=0
    )

    assert result == {'extractions': []}
