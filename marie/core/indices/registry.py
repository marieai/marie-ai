"""Index registry."""

from typing import Dict, Type

from marie.core.data_structs.struct_type import IndexStructType
from marie.core.indices.base import BaseIndex
from marie.core.indices.document_summary.base import DocumentSummaryIndex
from marie.core.indices.empty.base import EmptyIndex
from marie.core.indices.keyword_table.base import KeywordTableIndex
from marie.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from marie.core.indices.list.base import SummaryIndex
from marie.core.indices.multi_modal import MultiModalVectorStoreIndex
from marie.core.indices.property_graph import PropertyGraphIndex
from marie.core.indices.struct_store.pandas import PandasIndex
from marie.core.indices.struct_store.sql import SQLStructStoreIndex
from marie.core.indices.tree.base import TreeIndex
from marie.core.indices.vector_store.base import VectorStoreIndex

INDEX_STRUCT_TYPE_TO_INDEX_CLASS: Dict[IndexStructType, Type[BaseIndex]] = {
    IndexStructType.TREE: TreeIndex,
    IndexStructType.LIST: SummaryIndex,
    IndexStructType.KEYWORD_TABLE: KeywordTableIndex,
    IndexStructType.VECTOR_STORE: VectorStoreIndex,
    IndexStructType.SQL: SQLStructStoreIndex,
    IndexStructType.PANDAS: PandasIndex,  # type: ignore
    IndexStructType.KG: KnowledgeGraphIndex,
    IndexStructType.SIMPLE_LPG: PropertyGraphIndex,
    IndexStructType.EMPTY: EmptyIndex,
    IndexStructType.DOCUMENT_SUMMARY: DocumentSummaryIndex,
    IndexStructType.MULTIMODAL_VECTOR_STORE: MultiModalVectorStoreIndex,
}
