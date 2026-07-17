"""Trajectory middleware for indented call-stack visualization.

Provides a visual trace of agent execution with indentation showing
the call hierarchy.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TextIO

from marie.agent.middleware.protocol import BaseMiddleware

if TYPE_CHECKING:
    from marie.agent.emitter import Emitter


class TrajectoryMiddleware(BaseMiddleware):
    """Middleware that prints an indented call-stack trace.

    Visualizes the execution flow showing agent starts, tool calls,
    LLM calls, and their results with proper indentation.

    Example output:
        ```
        [agent.start] MyAgent
        │ [llm.start] gpt-4
        │ └─ [llm.success] 245ms
        │ [tool.start] search(query="hello")
        │ └─ [tool.success] 123ms
        └─ [agent.success] 500ms
        ```
    """

    def __init__(
        self,
        pretty: bool = True,
        output: Optional[TextIO] = None,
        show_timestamps: bool = False,
        max_arg_length: int = 50,
    ) -> None:
        """Initialize trajectory middleware.

        Args:
            pretty: Use tree-style formatting with Unicode characters
            output: Output stream (defaults to sys.stderr)
            show_timestamps: Include timestamps in output
            max_arg_length: Maximum length for argument display
        """
        super().__init__(name="TrajectoryMiddleware", priority=10)
        self.pretty = pretty
        self.output = output or sys.stderr
        self.show_timestamps = show_timestamps
        self.max_arg_length = max_arg_length
        self._depth = 0
        self._lines: List[str] = []

    def bind(self, emitter: "Emitter") -> None:
        """Bind trajectory visualization to emitter events."""
        # Agent events
        self._listener_ids.append(
            emitter.on(
                "agent.start",
                self._on_agent_start,
                priority=self.priority,
                is_blocking=True,
            )
        )
        self._listener_ids.append(
            emitter.on(
                "agent.success",
                self._on_agent_success,
                priority=self.priority,
                is_blocking=True,
            )
        )
        self._listener_ids.append(
            emitter.on(
                "agent.error",
                self._on_agent_error,
                priority=self.priority,
                is_blocking=True,
            )
        )

        # Tool events
        self._listener_ids.append(
            emitter.on(
                "tool.start",
                self._on_tool_start,
                priority=self.priority,
                is_blocking=True,
            )
        )
        self._listener_ids.append(
            emitter.on(
                "tool.success",
                self._on_tool_success,
                priority=self.priority,
                is_blocking=True,
            )
        )
        self._listener_ids.append(
            emitter.on(
                "tool.error",
                self._on_tool_error,
                priority=self.priority,
                is_blocking=True,
            )
        )

        # LLM events
        self._listener_ids.append(
            emitter.on(
                "llm.start",
                self._on_llm_start,
                priority=self.priority,
                is_blocking=True,
            )
        )
        self._listener_ids.append(
            emitter.on(
                "llm.success",
                self._on_llm_success,
                priority=self.priority,
                is_blocking=True,
            )
        )
        self._listener_ids.append(
            emitter.on(
                "llm.error",
                self._on_llm_error,
                priority=self.priority,
                is_blocking=True,
            )
        )

    def _prefix(self, is_end: bool = False) -> str:
        """Get the indentation prefix for current depth."""
        if not self.pretty:
            return "  " * self._depth

        if self._depth == 0:
            return ""

        prefix_parts = []
        for i in range(self._depth - 1):
            prefix_parts.append("│ ")

        if is_end:
            prefix_parts.append("└─ ")
        else:
            prefix_parts.append("│ ")

        return "".join(prefix_parts)

    def _write(self, message: str, is_end: bool = False) -> None:
        """Write a line to output with appropriate prefix."""
        line = f"{self._prefix(is_end)}{message}"
        self._lines.append(line)
        print(line, file=self.output)

    def _truncate(self, text: str, max_len: Optional[int] = None) -> str:
        """Truncate text to max length."""
        max_len = max_len or self.max_arg_length
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _format_args(self, args: Dict[str, Any]) -> str:
        """Format arguments for display."""
        if not args:
            return ""
        parts = []
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 20:
                v_str = v_str[:17] + "..."
            parts.append(f'{k}="{v_str}"')
        return self._truncate(", ".join(parts))

    # Agent event handlers
    def _on_agent_start(self, data: Dict[str, Any]) -> None:
        name = data.get("agent_name", "Agent")
        self._write(f"[agent.start] {name}")
        self._depth += 1

    def _on_agent_success(self, data: Dict[str, Any]) -> None:
        self._depth = max(0, self._depth - 1)
        duration = data.get("duration_ms", 0)
        self._write(f"[agent.success] {duration:.0f}ms", is_end=True)

    def _on_agent_error(self, data: Dict[str, Any]) -> None:
        self._depth = max(0, self._depth - 1)
        error = data.get("error_message", "Unknown error")
        self._write(f"[agent.error] {self._truncate(error)}", is_end=True)

    # Tool event handlers
    def _on_tool_start(self, data: Dict[str, Any]) -> None:
        name = data.get("tool_name", "tool")
        args = self._format_args(data.get("arguments", {}))
        self._write(f"[tool.start] {name}({args})")
        self._depth += 1

    def _on_tool_success(self, data: Dict[str, Any]) -> None:
        self._depth = max(0, self._depth - 1)
        duration = data.get("duration_ms", 0)
        self._write(f"[tool.success] {duration:.0f}ms", is_end=True)

    def _on_tool_error(self, data: Dict[str, Any]) -> None:
        self._depth = max(0, self._depth - 1)
        error = data.get("error", "Unknown error")
        self._write(f"[tool.error] {self._truncate(error)}", is_end=True)

    # LLM event handlers
    def _on_llm_start(self, data: Dict[str, Any]) -> None:
        model = data.get("model_name", "llm")
        self._write(f"[llm.start] {model}")
        self._depth += 1

    def _on_llm_success(self, data: Dict[str, Any]) -> None:
        self._depth = max(0, self._depth - 1)
        duration = data.get("duration_ms", 0)
        has_tools = data.get("has_tool_calls", False)
        suffix = " (tool_calls)" if has_tools else ""
        self._write(f"[llm.success] {duration:.0f}ms{suffix}", is_end=True)

    def _on_llm_error(self, data: Dict[str, Any]) -> None:
        self._depth = max(0, self._depth - 1)
        error = data.get("error_message", "Unknown error")
        self._write(f"[llm.error] {self._truncate(error)}", is_end=True)

    def get_trace(self) -> str:
        """Get the accumulated trace as a string."""
        return "\n".join(self._lines)

    def clear(self) -> None:
        """Clear the accumulated trace."""
        self._lines.clear()
        self._depth = 0
