"""System info tool for agent framework."""

from __future__ import annotations

import json
import os
import platform
import sys
from typing import Any

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput


class SystemInfoTool(AgentTool):
    """Get system and environment information.

    Returns Python version, OS, machine architecture, CWD, user, and environment summary.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="system_info",
            description="Get system and environment information (Python version, OS, machine, CWD, user).",
        )

    def call(self, **kwargs: Any) -> ToolOutput:
        """Get system information.

        Returns:
            ToolOutput with system information
        """
        info = {
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "cwd": os.getcwd(),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "home": os.environ.get("HOME", os.environ.get("USERPROFILE", "unknown")),
            "env_var_count": len(os.environ),
        }

        return ToolOutput(
            content=json.dumps(info, indent=2),
            tool_name=self.name,
            raw_input=kwargs,
            raw_output=info,
            is_error=False,
        )
