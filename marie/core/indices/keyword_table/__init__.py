"""Keyword Table Index Data Structures."""

# indices
from marie.core.indices.keyword_table.base import (
    GPTKeywordTableIndex,
    KeywordTableIndex,
)
from marie.core.indices.keyword_table.rake_base import (
    GPTRAKEKeywordTableIndex,
    RAKEKeywordTableIndex,
)
from marie.core.indices.keyword_table.retrievers import (
    KeywordTableGPTRetriever,
    KeywordTableRAKERetriever,
    KeywordTableSimpleRetriever,
)
from marie.core.indices.keyword_table.simple_base import (
    GPTSimpleKeywordTableIndex,
    SimpleKeywordTableIndex,
)

__all__ = [
    "KeywordTableIndex",
    "SimpleKeywordTableIndex",
    "RAKEKeywordTableIndex",
    "KeywordTableGPTRetriever",
    "KeywordTableRAKERetriever",
    "KeywordTableSimpleRetriever",
    # legacy
    "GPTKeywordTableIndex",
    "GPTSimpleKeywordTableIndex",
    "GPTRAKEKeywordTableIndex",
]
