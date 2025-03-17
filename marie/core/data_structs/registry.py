"""Index registry."""

from typing import Dict, Type

from marie.core.data_structs.data_structs import (
    KG,
    EmptyIndexStruct,
    IndexDict,
    IndexGraph,
    IndexList,
    IndexLPG,
    IndexStruct,
    KeywordTable,
    MultiModelIndexDict,
)
from marie.core.data_structs.document_summary import IndexDocumentSummary
from marie.core.data_structs.struct_type import IndexStructType
from marie.core.data_structs.table import PandasStructTable, SQLStructTable

INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS: Dict[IndexStructType, Type[IndexStruct]] = {
    IndexStructType.TREE: IndexGraph,
    IndexStructType.LIST: IndexList,
    IndexStructType.KEYWORD_TABLE: KeywordTable,
    IndexStructType.VECTOR_STORE: IndexDict,
    IndexStructType.SQL: SQLStructTable,
    IndexStructType.PANDAS: PandasStructTable,
    IndexStructType.KG: KG,
    IndexStructType.SIMPLE_LPG: IndexLPG,
    IndexStructType.EMPTY: EmptyIndexStruct,
    IndexStructType.DOCUMENT_SUMMARY: IndexDocumentSummary,
    IndexStructType.MULTIMODAL_VECTOR_STORE: MultiModelIndexDict,
}
