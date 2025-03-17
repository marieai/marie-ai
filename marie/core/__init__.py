"""Init file of LlamaIndex."""

__version__ = "0.12.14"

import logging
from logging import NullHandler
from typing import Callable, Optional

try:
    # Force pants to install eval_type_backport on 3.9
    import eval_type_backport  # noqa  # type: ignore
except ImportError:
    pass

# response
from marie.core.base.response.schema import Response

# import global eval handler
from marie.core.callbacks.global_handlers import set_global_handler
from marie.core.data_structs.struct_type import IndexStructType
from marie.core.embeddings.mock_embed_model import MockEmbedding

# indices
# loading
from marie.core.indices import (
    ComposableGraph,
    DocumentSummaryIndex,
    GPTDocumentSummaryIndex,
    GPTKeywordTableIndex,
    GPTListIndex,
    GPTRAKEKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
    GPTTreeIndex,
    GPTVectorStoreIndex,
    KeywordTableIndex,
    KnowledgeGraphIndex,
    ListIndex,
    PropertyGraphIndex,
    RAKEKeywordTableIndex,
    SimpleKeywordTableIndex,
    SummaryIndex,
    TreeIndex,
    VectorStoreIndex,
    load_graph_from_storage,
    load_index_from_storage,
    load_indices_from_storage,
)

# structured
from marie.core.indices.common.struct_store.base import (
    SQLDocumentContextBuilder,
)

# prompt helper
from marie.core.indices.prompt_helper import PromptHelper

# prompts
from marie.core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    # backwards compatibility
    Prompt,
    PromptTemplate,
    SelectorPromptTemplate,
)
from marie.core.readers import SimpleDirectoryReader, download_loader

# Response Synthesizer
from marie.core.response_synthesizers.factory import get_response_synthesizer
from marie.core.schema import Document, QueryBundle
from marie.core.service_context import (
    ServiceContext,
    set_global_service_context,
)

# global settings
from marie.core.settings import Settings

# storage
from marie.core.storage.storage_context import StorageContext

# sql wrapper
from marie.core.utilities.sql_wrapper import SQLDatabase

# global tokenizer
from marie.core.utils import get_tokenizer, set_global_tokenizer

# best practices for library logging:
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(__name__).addHandler(NullHandler())

__all__ = [
    "StorageContext",
    "ServiceContext",
    "ComposableGraph",
    # indices
    "SummaryIndex",
    "VectorStoreIndex",
    "SimpleKeywordTableIndex",
    "KeywordTableIndex",
    "RAKEKeywordTableIndex",
    "TreeIndex",
    "DocumentSummaryIndex",
    "KnowledgeGraphIndex",
    "PropertyGraphIndex",
    # indices - legacy names
    "GPTKeywordTableIndex",
    "GPTKnowledgeGraphIndex",
    "GPTSimpleKeywordTableIndex",
    "GPTRAKEKeywordTableIndex",
    "GPTListIndex",
    "ListIndex",
    "GPTTreeIndex",
    "GPTVectorStoreIndex",
    "GPTDocumentSummaryIndex",
    "Prompt",
    "PromptTemplate",
    "BasePromptTemplate",
    "ChatPromptTemplate",
    "SelectorPromptTemplate",
    "SummaryPrompt",
    "TreeInsertPrompt",
    "TreeSelectPrompt",
    "TreeSelectMultiplePrompt",
    "RefinePrompt",
    "QuestionAnswerPrompt",
    "KeywordExtractPrompt",
    "QueryKeywordExtractPrompt",
    "Response",
    "Document",
    "SimpleDirectoryReader",
    "VellumPredictor",
    "VellumPromptRegistry",
    "MockEmbedding",
    "SQLDatabase",
    "SQLDocumentContextBuilder",
    "SQLContextBuilder",
    "PromptHelper",
    "IndexStructType",
    "download_loader",
    "load_graph_from_storage",
    "load_index_from_storage",
    "load_indices_from_storage",
    "QueryBundle",
    "get_response_synthesizer",
    "set_global_service_context",
    "set_global_handler",
    "set_global_tokenizer",
    "get_tokenizer",
    "Settings",
]

# eval global toggle
from marie.core.callbacks.base_handler import BaseCallbackHandler

global_handler: Optional[BaseCallbackHandler] = None

# NOTE: keep for backwards compatibility
SQLContextBuilder = SQLDocumentContextBuilder

# global tokenizer
global_tokenizer: Optional[Callable[[str], list]] = None
