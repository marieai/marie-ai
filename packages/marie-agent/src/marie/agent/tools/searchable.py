"""Searchable toolset for dynamic tool discovery.

This module provides BM25-based tool discovery inspired by Haystack's
SearchableToolset pattern. Instead of exposing all tools to the LLM,
it exposes a `search_tools` meta-function that lets the model discover
relevant tools dynamically.

Usage:
    ```python
    from marie.agent import ReactAgent
    from marie.agent.tools import SearchableToolset

    # Create a searchable toolset
    toolset = SearchableToolset(
        tools=["calculator", "weather", "search", ...],  # 100+ tools
        passthrough_threshold=5,
        top_k=3,
    )

    # Pass to agent - clean, no extra parameters
    agent = ReactAgent(
        llm=llm,
        tools=toolset,
    )
    ```

See: https://haystack.deepset.ai/release-notes/2.25.0
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, FunctionTool

logger = logging.getLogger(__name__)

# Lazy import bm25s to avoid import errors if not installed
_bm25s = None


def _get_bm25s():
    """Lazy import bm25s module."""
    global _bm25s
    if _bm25s is None:
        try:
            import bm25s

            _bm25s = bm25s
        except ImportError:
            logger.warning(
                "bm25s not installed. Install with: uv add 'bm25s[core]>=0.2.0'"
            )
            _bm25s = False
    return _bm25s if _bm25s else None


class SearchToolsInput(BaseModel):
    """Input schema for the search_tools meta-function."""

    query: str = Field(
        ...,
        description="Natural language query describing the tool capability needed",
    )


class SearchableToolset:
    """BM25-based dynamic tool discovery.

    Wraps a collection of tools and exposes them through a `search_tools`
    meta-function. This reduces prompt size and improves accuracy for
    large tool catalogs (100+ tools).

    When the catalog is small (<= passthrough_threshold), all tools are
    exposed directly without the search layer.

    Example:
        ```python
        from marie.agent import ReactAgent
        from marie.agent.tools import SearchableToolset

        # Create searchable toolset from tool catalog
        toolset = SearchableToolset(
            tools=["calculator", "weather", "search", my_custom_tool, ...],
            passthrough_threshold=5,
            top_k=3,
        )

        # Pass directly to agent
        agent = ReactAgent(
            llm=llm,
            tools=toolset,  # Clean API - no extra params needed
        )

        # Run agent - tools discovered dynamically
        for responses in agent.run(messages):
            print(responses[-1].content)
        ```

    The agent will:
    1. Expose only `search_tools` to the LLM (if > passthrough_threshold tools)
    2. When LLM calls `search_tools("calculate math")`, matching tools are found
    3. Found tools are registered and available for the next LLM call
    4. Tools are automatically cleaned up after the request
    """

    def __init__(
        self,
        tools: List[Union[str, Dict, AgentTool, Callable]],
        passthrough_threshold: int = 5,
        top_k: int = 3,
    ):
        """Initialize the searchable toolset.

        Args:
            tools: Tool catalog. Accepts same formats as agent's function_list:
                - Tool name strings (looked up from registry)
                - Configuration dicts with 'name' key
                - AgentTool instances
                - Callable functions
            passthrough_threshold: If tool count <= this, expose all directly
            top_k: Maximum tools to return per search
        """
        self._tool_specs = list(tools)
        self._passthrough_threshold = passthrough_threshold
        self._top_k = top_k

        # Resolved tools (populated when bound to agent)
        self._tools: List[AgentTool] = []
        self._resolved = False

        # BM25 index state
        self._retriever = None
        self._indexed = False

        # Agent binding (set when attached to an agent)
        self._register_callback: Optional[Callable[[AgentTool], None]] = None
        self._tools_dirty_callback: Optional[Callable[[], None]] = None

        # Track dynamically registered tools for cleanup
        self._dynamically_added_tools: List[str] = []

    def _resolve_tools(self) -> None:
        """Resolve tool specifications to AgentTool instances."""
        if self._resolved:
            return

        from marie.agent.tools.registry import resolve_tools

        resolved = resolve_tools(self._tool_specs)
        self._tools = list(resolved.values())
        self._resolved = True

        # Build index now that we have tools
        if self._tools:
            self._build_index()

    def bind(
        self,
        register_callback: Callable[[AgentTool], None],
        tools_dirty_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Bind toolset to an agent.

        Called by the agent when the toolset is attached. Sets up callbacks
        for dynamic tool registration.

        Args:
            register_callback: Function to register discovered tools
            tools_dirty_callback: Function to signal tools have changed
        """
        self._register_callback = register_callback
        self._tools_dirty_callback = tools_dirty_callback

        # Resolve tools now
        self._resolve_tools()

    @property
    def is_bound(self) -> bool:
        """Check if toolset is bound to an agent."""
        return self._register_callback is not None

    @property
    def is_passthrough(self) -> bool:
        """Check if operating in passthrough mode (small catalog)."""
        self._resolve_tools()
        return len(self._tools) <= self._passthrough_threshold

    @property
    def tool_count(self) -> int:
        """Number of tools in the catalog."""
        self._resolve_tools()
        return len(self._tools)

    @property
    def tools(self) -> List[AgentTool]:
        """Get all tools in the catalog."""
        self._resolve_tools()
        return list(self._tools)

    def _build_index(self) -> None:
        """Build BM25 index from tool metadata."""
        bm25s = _get_bm25s()
        if bm25s is None:
            logger.warning("BM25 unavailable, falling back to linear search")
            self._indexed = False
            return

        if not self._tools:
            self._indexed = False
            return

        # Build corpus from tool name, description, and parameter names
        corpus = []
        for tool in self._tools:
            params_dict = tool.metadata.get_parameters_dict()
            param_names = list(params_dict.get("properties", {}).keys())

            text = (
                f"{tool.metadata.name} "
                f"{tool.metadata.description} "
                f"{' '.join(param_names)}"
            )
            corpus.append(text)

        try:
            corpus_tokens = bm25s.tokenize(corpus, lower=True)
            self._retriever = bm25s.BM25()
            self._retriever.index(corpus_tokens)
            self._indexed = True
            logger.debug(f"Built BM25 index for {len(self._tools)} tools")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
            self._indexed = False

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[AgentTool, float]]:
        """Search for tools matching the query.

        Args:
            query: Natural language query
            top_k: Maximum results (defaults to instance top_k)

        Returns:
            List of (tool, score) tuples sorted by relevance
        """
        self._resolve_tools()

        if not self._tools:
            return []

        k = top_k or self._top_k

        if not self._indexed or self._retriever is None:
            return self._fallback_search(query, k)

        bm25s = _get_bm25s()
        if bm25s is None:
            return self._fallback_search(query, k)

        try:
            query_tokens = bm25s.tokenize([query], lower=True)

            # Get ALL results, then take top-k
            num_tools = len(self._tools)
            results, scores = self._retriever.retrieve(query_tokens, k=num_tools)

            indices = results[0]
            score_values = scores[0]

            ranked = [
                (self._tools[idx], float(score))
                for idx, score in zip(indices, score_values)
            ]

            return ranked[:k]

        except Exception as e:
            logger.warning(f"BM25 search failed: {e}, falling back to linear")
            return self._fallback_search(query, k)

    def _fallback_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[AgentTool, float]]:
        """Fallback keyword-based search when BM25 unavailable."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        results = []
        for tool in self._tools:
            tool_text = f"{tool.metadata.name} {tool.metadata.description}".lower()
            tool_terms = set(tool_text.split())

            overlap = len(query_terms & tool_terms)
            if overlap > 0:
                score = overlap / len(query_terms)
                results.append((tool, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _search_tools_fn(self, query: str) -> str:
        """The search_tools function exposed to the LLM."""
        results = self.search(query)

        if not results:
            return json.dumps(
                {
                    "found": 0,
                    "tools": [],
                    "message": "No matching tools found. Try a different query.",
                }
            )

        tool_infos = []
        for tool, score in results:
            # Register tool for execution
            if self._register_callback:
                self._register_callback(tool)
                self._dynamically_added_tools.append(tool.metadata.name)

            tool_infos.append(
                {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.get_parameters_dict(),
                    "relevance_score": round(score, 3),
                }
            )

        # Signal that tools have changed
        if self._tools_dirty_callback:
            self._tools_dirty_callback()

        return json.dumps(
            {
                "found": len(tool_infos),
                "tools": tool_infos,
                "message": f"Found {len(tool_infos)} relevant tools. You can now call them by name.",
            }
        )

    def get_search_tool(self) -> AgentTool:
        """Get the search_tools meta-function as an AgentTool."""
        return FunctionTool.from_defaults(
            fn=self._search_tools_fn,
            name="search_tools",
            description=(
                "Search for available tools by describing what capability you need. "
                "Returns matching tools that you can then call. Use this when you need "
                "a tool but don't know which one to use."
            ),
            fn_schema=SearchToolsInput,
        )

    def get_exposed_tools(self) -> List[AgentTool]:
        """Get tools to expose to the LLM.

        In passthrough mode, returns all tools directly.
        Otherwise, returns only the search_tools meta-function.
        """
        self._resolve_tools()

        if self.is_passthrough:
            return list(self._tools)
        else:
            return [self.get_search_tool()]

    def get_all_tools(self) -> Dict[str, AgentTool]:
        """Get all tools as a dict (for function_map initialization)."""
        self._resolve_tools()
        return {tool.metadata.name: tool for tool in self._tools}

    def get_dynamically_added_tools(self) -> List[str]:
        """Get names of tools dynamically added via search."""
        return list(self._dynamically_added_tools)

    def clear_dynamic_tools(self) -> None:
        """Clear the list of dynamically added tools."""
        self._dynamically_added_tools = []

    def get_tool(self, name: str) -> Optional[AgentTool]:
        """Get a tool by name."""
        self._resolve_tools()
        for tool in self._tools:
            if tool.metadata.name == name:
                return tool
        return None
