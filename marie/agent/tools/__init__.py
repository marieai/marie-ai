"""Marie agent tools module."""

from marie.agent.tools.base import (
    AgentTool,
    FunctionTool,
    ToolMetadata,
    ToolOutput,
    adapt_tool,
)

# Database tools (asyncpg-based)
from marie.agent.tools.database import (
    AsyncDatabaseTool,
    MemoryTool,
    NotesTool,
    PostgresTool,
    TodoTool,
)

# RAG tools
from marie.agent.tools.document_search import (
    DocumentSearchTool,
    MultiDocumentSearchTool,
)

# Filesystem tools
from marie.agent.tools.filesystem import (
    FileListInput,
    FileListTool,
    FileReadInput,
    FileReadTool,
    FileWriteInput,
    FileWriteTool,
    ShellInput,
    ShellTool,
)

# Query Plan tools (auto-registered via @register_tool decorator)
from marie.agent.tools.query_plan_tools import (
    get_node_defaults,
    get_node_types,
    validate_query_plan,
)
from marie.agent.tools.rag_tool import (
    RAGDeleteInput,
    RAGIngestInput,
    RAGIngestTool,
    RAGSearchInput,
    RAGSearchTool,
    RAGStatsInput,
    RAGTool,
)
from marie.agent.tools.registry import (
    TOOL_REGISTRY,
    get_tool,
    list_tools,
    register_tool,
    resolve_tools,
)

# Utility tools
from marie.agent.tools.utilities import (
    HttpRequestInput,
    HttpRequestTool,
    SystemInfoTool,
    WebFetchInput,
    WebFetchTool,
)
from marie.agent.tools.wrappers import ComponentTool

__all__ = [
    # Base classes
    "AgentTool",
    "FunctionTool",
    "ToolMetadata",
    "ToolOutput",
    "adapt_tool",
    # Component tools
    "ComponentTool",
    # Registry
    "TOOL_REGISTRY",
    "register_tool",
    "get_tool",
    "list_tools",
    "resolve_tools",
    # Database tools
    "AsyncDatabaseTool",
    "MemoryTool",
    "NotesTool",
    "PostgresTool",
    "TodoTool",
    # RAG tools
    "DocumentSearchTool",
    "MultiDocumentSearchTool",
    "RAGTool",
    "RAGSearchTool",
    "RAGIngestTool",
    "RAGSearchInput",
    "RAGIngestInput",
    "RAGDeleteInput",
    "RAGStatsInput",
    # Filesystem tools
    "FileListInput",
    "FileListTool",
    "FileReadInput",
    "FileReadTool",
    "FileWriteInput",
    "FileWriteTool",
    "ShellInput",
    "ShellTool",
    # Utility tools
    "HttpRequestInput",
    "HttpRequestTool",
    "SystemInfoTool",
    "WebFetchInput",
    "WebFetchTool",
    # Query Plan tools
    "get_node_types",
    "validate_query_plan",
    "get_node_defaults",
]
