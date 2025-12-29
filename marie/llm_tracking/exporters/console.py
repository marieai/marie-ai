"""
Console Exporter - Development exporter that logs to stdout.

Useful for debugging and local development without external dependencies.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, TextIO

from marie.llm_tracking.config import get_settings
from marie.llm_tracking.exporters.base import AbstractExporter
from marie.llm_tracking.types import Observation, Score, Trace

logger = logging.getLogger(__name__)


class ConsoleExporter(AbstractExporter):
    """
    Exporter that outputs events to console/stdout.

    Features:
    - Pretty-printed JSON output
    - Colorized output (when terminal supports it)
    - Configurable verbosity
    - Optional file output
    """

    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
    }

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        pretty: bool = True,
        colorize: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the console exporter.

        Args:
            stream: Output stream (defaults to sys.stdout)
            pretty: Whether to pretty-print JSON
            colorize: Whether to use ANSI colors
            verbose: Whether to include full payloads
        """
        self._stream = stream or sys.stdout
        self._pretty = pretty
        self._colorize = colorize and self._stream.isatty()
        self._verbose = verbose
        self._settings = get_settings()

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colorization is enabled."""
        if not self._colorize:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _format_timestamp(self, dt: datetime) -> str:
        """Format a timestamp for display."""
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _truncate(self, text: str, max_length: int = 200) -> str:
        """Truncate text for display."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def _format_data(self, data: Any) -> str:
        """Format data for display."""
        if data is None:
            return self._color("null", "dim")

        if isinstance(data, str):
            if not self._verbose:
                data = self._truncate(data)
            return data

        try:
            if self._pretty:
                text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            else:
                text = json.dumps(data, ensure_ascii=False, default=str)

            if not self._verbose:
                text = self._truncate(text)
            return text
        except (TypeError, ValueError):
            return str(data)

    def _print(self, message: str) -> None:
        """Print a message to the output stream."""
        try:
            print(message, file=self._stream)
            self._stream.flush()
        except Exception as e:
            logger.error(f"Console exporter failed to write: {e}")

    def export_trace(self, trace: Trace) -> None:
        """Export a trace to console."""
        header = self._color("═══ TRACE ═══", "bold")
        timestamp = self._color(self._format_timestamp(trace.timestamp), "dim")

        lines = [
            f"\n{header}",
            f"  {self._color('ID:', 'cyan')} {trace.id}",
            f"  {self._color('Name:', 'cyan')} {trace.name or '(unnamed)'}",
            f"  {self._color('Timestamp:', 'cyan')} {timestamp}",
        ]

        if trace.user_id:
            lines.append(f"  {self._color('User:', 'cyan')} {trace.user_id}")
        if trace.session_id:
            lines.append(f"  {self._color('Session:', 'cyan')} {trace.session_id}")
        if trace.tags:
            lines.append(f"  {self._color('Tags:', 'cyan')} {', '.join(trace.tags)}")

        if self._verbose:
            if trace.input is not None:
                lines.append(f"  {self._color('Input:', 'cyan')}")
                lines.append(f"    {self._format_data(trace.input)}")
            if trace.output is not None:
                lines.append(f"  {self._color('Output:', 'cyan')}")
                lines.append(f"    {self._format_data(trace.output)}")
            if trace.metadata:
                lines.append(f"  {self._color('Metadata:', 'cyan')}")
                lines.append(f"    {self._format_data(trace.metadata)}")

        self._print("\n".join(lines))

    def export_observation(self, observation: Observation) -> None:
        """Export an observation to console."""
        type_colors = {
            "GENERATION": "green",
            "SPAN": "blue",
            "EVENT": "yellow",
        }
        type_color = type_colors.get(observation.type.value, "reset")

        header = self._color(f"─── {observation.type.value} ───", type_color)
        timestamp = self._color(self._format_timestamp(observation.start_time), "dim")

        lines = [
            f"\n{header}",
            f"  {self._color('ID:', 'cyan')} {observation.id}",
            f"  {self._color('Trace:', 'cyan')} {observation.trace_id}",
            f"  {self._color('Name:', 'cyan')} {observation.name or '(unnamed)'}",
            f"  {self._color('Start:', 'cyan')} {timestamp}",
        ]

        if observation.end_time:
            duration_ms = (
                observation.end_time - observation.start_time
            ).total_seconds() * 1000
            duration_str = f"{duration_ms:.2f}ms"
            lines.append(f"  {self._color('Duration:', 'cyan')} {duration_str}")

        if observation.model:
            lines.append(f"  {self._color('Model:', 'cyan')} {observation.model}")

        if observation.usage:
            usage_parts = []
            if observation.usage.input_tokens is not None:
                usage_parts.append(f"in={observation.usage.input_tokens}")
            if observation.usage.output_tokens is not None:
                usage_parts.append(f"out={observation.usage.output_tokens}")
            if observation.usage.total_tokens is not None:
                usage_parts.append(f"total={observation.usage.total_tokens}")
            if usage_parts:
                lines.append(
                    f"  {self._color('Tokens:', 'cyan')} {', '.join(usage_parts)}"
                )

        if observation.cost and observation.cost.total_cost:
            cost_str = f"${float(observation.cost.total_cost):.6f}"
            lines.append(f"  {self._color('Cost:', 'cyan')} {cost_str}")

        if observation.level.value != "DEFAULT":
            level_colors = {
                "DEBUG": "dim",
                "WARNING": "yellow",
                "ERROR": "red",
            }
            level_color = level_colors.get(observation.level.value, "reset")
            lines.append(
                f"  {self._color('Level:', 'cyan')} {self._color(observation.level.value, level_color)}"
            )

        if observation.status_message:
            lines.append(
                f"  {self._color('Status:', 'cyan')} {observation.status_message}"
            )

        if self._verbose:
            if observation.input is not None:
                lines.append(f"  {self._color('Input:', 'cyan')}")
                lines.append(f"    {self._format_data(observation.input)}")
            if observation.output is not None:
                lines.append(f"  {self._color('Output:', 'cyan')}")
                lines.append(f"    {self._format_data(observation.output)}")

        self._print("\n".join(lines))

    def export_score(self, score: Score) -> None:
        """Export a score to console."""
        header = self._color("★ SCORE ★", "magenta")
        timestamp = self._color(self._format_timestamp(score.timestamp), "dim")

        lines = [
            f"\n{header}",
            f"  {self._color('ID:', 'cyan')} {score.id}",
            f"  {self._color('Trace:', 'cyan')} {score.trace_id}",
            f"  {self._color('Name:', 'cyan')} {score.name}",
            f"  {self._color('Value:', 'cyan')} {score.value} ({score.data_type.value})",
            f"  {self._color('Source:', 'cyan')} {score.source}",
            f"  {self._color('Timestamp:', 'cyan')} {timestamp}",
        ]

        if score.observation_id:
            lines.append(
                f"  {self._color('Observation:', 'cyan')} {score.observation_id}"
            )
        if score.comment:
            lines.append(f"  {self._color('Comment:', 'cyan')} {score.comment}")

        self._print("\n".join(lines))

    def flush(self) -> None:
        """Flush the output stream."""
        try:
            self._stream.flush()
        except Exception:
            pass
