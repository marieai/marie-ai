"""LlamaIndex data structures."""

# indices
from marie.core.indices.composability.graph import ComposableGraph
from marie.core.indices.document_summary import (
    DocumentSummaryIndex,
    GPTDocumentSummaryIndex,
)
from marie.core.indices.document_summary.base import DocumentSummaryIndex
from marie.core.indices.empty.base import EmptyIndex, GPTEmptyIndex
from marie.core.indices.keyword_table.base import (
    GPTKeywordTableIndex,
    KeywordTableIndex,
)
from marie.core.indices.keyword_table.rake_base import (
    GPTRAKEKeywordTableIndex,
    RAKEKeywordTableIndex,
)
from marie.core.indices.keyword_table.simple_base import (
    GPTSimpleKeywordTableIndex,
    SimpleKeywordTableIndex,
)
from marie.core.indices.knowledge_graph import (
    KnowledgeGraphIndex,
)
from marie.core.indices.list import GPTListIndex, ListIndex, SummaryIndex
from marie.core.indices.list.base import (
    GPTListIndex,
    ListIndex,
    SummaryIndex,
)
from marie.core.indices.loading import (
    load_graph_from_storage,
    load_index_from_storage,
    load_indices_from_storage,
)
from marie.core.indices.multi_modal import MultiModalVectorStoreIndex
from marie.core.indices.struct_store.pandas import (
    GPTPandasIndex,
    PandasIndex,
)
from marie.core.indices.struct_store.sql import (
    GPTSQLStructStoreIndex,
    SQLStructStoreIndex,
)
from marie.core.indices.tree.base import GPTTreeIndex, TreeIndex
from marie.core.indices.vector_store import (
    GPTVectorStoreIndex,
    VectorStoreIndex,
)

from marie.core.indices.property_graph.base import (
    PropertyGraphIndex,
)

__all__ = [
    "load_graph_from_storage",
    "load_index_from_storage",
    "load_indices_from_storage",
    "KeywordTableIndex",
    "SimpleKeywordTableIndex",
    "RAKEKeywordTableIndex",
    "SummaryIndex",
    "TreeIndex",
    "DocumentSummaryIndex",
    "KnowledgeGraphIndex",
    "PandasIndex",
    "VectorStoreIndex",
    "SQLStructStoreIndex",
    "MultiModalVectorStoreIndex",
    "EmptyIndex",
    "ComposableGraph",
    "PropertyGraphIndex",
    # legacy
    "GPTKnowledgeGraphIndex",
    "GPTKeywordTableIndex",
    "GPTSimpleKeywordTableIndex",
    "GPTRAKEKeywordTableIndex",
    "GPTDocumentSummaryIndex",
    "GPTListIndex",
    "GPTTreeIndex",
    "GPTPandasIndex",
    "ListIndex",
    "GPTVectorStoreIndex",
    "GPTSQLStructStoreIndex",
    "GPTEmptyIndex",
]
