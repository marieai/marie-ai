"""
Type definitions for marie-wasm.

This module contains all data classes and enums used throughout the package.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Language(str, Enum):
    """Supported programming languages for Wasm compilation."""

    RUST = "rust"
    PYTHON = "python"
    JAVASCRIPT = "js"

    @classmethod
    def from_string(cls, value: str) -> "Language":
        """Parse language from string, case-insensitive."""
        normalized = value.lower().strip()
        # Handle common aliases
        aliases = {
            "javascript": cls.JAVASCRIPT,
            "js": cls.JAVASCRIPT,
            "typescript": cls.JAVASCRIPT,  # Compiled via esbuild
            "ts": cls.JAVASCRIPT,
            "python": cls.PYTHON,
            "py": cls.PYTHON,
            "rust": cls.RUST,
            "rs": cls.RUST,
        }
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(f"Unsupported language: {value}")


@dataclass
class CompilerConfig:
    """Configuration for a language compiler."""

    # Primary source filename expected in workspace
    source_filename: str
    # Additional files that can be included (dependencies, configs)
    additional_files: list[str] = field(default_factory=list)
    # Docker image suffix (e.g., "rust" -> "marie-compiler-rust")
    image_suffix: str = ""

    def __post_init__(self) -> None:
        if not self.image_suffix:
            # Default to inferring from source extension
            ext_map = {
                ".rs": "rust",
                ".py": "python",
                ".js": "js",
                ".ts": "js",
            }
            for ext, suffix in ext_map.items():
                if self.source_filename.endswith(ext):
                    self.image_suffix = suffix
                    break


# Pre-defined compiler configurations for each language
COMPILER_CONFIGS: dict[Language, CompilerConfig] = {
    Language.RUST: CompilerConfig(
        source_filename="lib.rs",
        additional_files=["Cargo.toml"],
        image_suffix="rust",
    ),
    Language.PYTHON: CompilerConfig(
        source_filename="main.py",
        additional_files=[],
        image_suffix="python",
    ),
    Language.JAVASCRIPT: CompilerConfig(
        source_filename="main.js",
        additional_files=["package.json"],
        image_suffix="js",
    ),
}


@dataclass
class Permissions:
    """
    Capability-based permissions for Wasm execution.

    These permissions control what host functions a Wasm module can access.
    The principle of least privilege applies - deny by default.
    """

    # HTTP client permissions
    allow_http: bool = False
    http_allowed_hosts: list[str] = field(default_factory=list)

    # Secrets access permissions
    allow_secrets: bool = False
    secret_allowed_names: list[str] = field(default_factory=list)

    # Key-value store permissions
    allow_kv: bool = False
    kv_prefix: str = ""  # Keys are prefixed to isolate node data

    # Resource limits
    max_memory_mb: int = 64
    max_fuel: int = 1_000_000_000  # CPU instruction budget (~1 billion)
    timeout_ms: int = 30_000  # 30 second default timeout

    def is_host_allowed(self, host: str) -> bool:
        """Check if a host is in the allowed list."""
        if not self.allow_http:
            return False
        if not self.http_allowed_hosts:
            return True  # Empty list means all hosts allowed
        return any(
            host == allowed or host.endswith(f".{allowed}")
            for allowed in self.http_allowed_hosts
        )

    def is_secret_allowed(self, name: str) -> bool:
        """Check if a secret name is in the allowed list."""
        if not self.allow_secrets:
            return False
        if not self.secret_allowed_names:
            return True  # Empty list means all secrets allowed
        return name in self.secret_allowed_names


@dataclass
class ExecutionContext:
    """
    Context passed to node execution.

    This provides the node with information about the current
    workflow execution for logging, correlation, and state management.
    """

    workflow_id: str
    execution_id: str
    node_id: str
    run_index: int = 0  # For retries or loop iterations

    def to_dict(self) -> dict[str, str | int]:
        """Convert to dict for WIT component model."""
        return {
            "workflow-id": self.workflow_id,
            "execution-id": self.execution_id,
            "node-id": self.node_id,
            "run-index": self.run_index,
        }


@dataclass
class DataItem:
    """
    Input/output data item for node execution.

    Mirrors the WIT data-item record. JSON is the primary format,
    with optional binary data for files/images.
    """

    json_data: str
    binary: Optional[bytes] = None

    def to_dict(self) -> dict[str, str | bytes | None]:
        """Convert to dict for WIT component model."""
        result: dict[str, str | bytes | None] = {"json": self.json_data}
        if self.binary is not None:
            result["binary"] = self.binary
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "DataItem":
        """Create from WIT component model dict."""
        return cls(
            json_data=data.get("json", "{}"),
            binary=data.get("binary"),
        )


@dataclass
class ExecutionResult:
    """
    Result of node execution.

    Either success with output data items, or failure with error message.
    Also tracks resource consumption for monitoring.
    """

    success: bool
    data: list[DataItem] = field(default_factory=list)
    error: Optional[str] = None
    fuel_consumed: int = 0
    execution_time_ms: float = 0.0

    @classmethod
    def ok(cls, data: list[DataItem], fuel_consumed: int = 0) -> "ExecutionResult":
        """Create successful result."""
        return cls(success=True, data=data, fuel_consumed=fuel_consumed)

    @classmethod
    def err(cls, error: str, fuel_consumed: int = 0) -> "ExecutionResult":
        """Create error result."""
        return cls(success=False, error=error, fuel_consumed=fuel_consumed)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        if self.success:
            return {
                "success": True,
                "data": [item.to_dict() for item in self.data],
                "fuel_consumed": self.fuel_consumed,
                "execution_time_ms": self.execution_time_ms,
            }
        else:
            return {
                "success": False,
                "error": self.error,
                "fuel_consumed": self.fuel_consumed,
                "execution_time_ms": self.execution_time_ms,
            }


@dataclass
class CompileRequest:
    """Request to compile code to Wasm."""

    code: str
    language: Language
    node_id: str
    dependencies: Optional[dict[str, str]] = None
    timeout_seconds: int = 120


@dataclass
class CompileResponse:
    """Response from compilation."""

    success: bool
    wasm_path: Optional[str] = None
    error: Optional[str] = None
    compile_time_ms: float = 0.0
    wasm_size_bytes: int = 0

    @classmethod
    def ok(
        cls, wasm_path: str, compile_time_ms: float = 0.0, wasm_size_bytes: int = 0
    ) -> "CompileResponse":
        """Create successful response."""
        return cls(
            success=True,
            wasm_path=wasm_path,
            compile_time_ms=compile_time_ms,
            wasm_size_bytes=wasm_size_bytes,
        )

    @classmethod
    def err(cls, error: str) -> "CompileResponse":
        """Create error response."""
        return cls(success=False, error=error)


# Built-in node permission presets
BUILTIN_PERMISSIONS: dict[str, Permissions] = {
    "http-request": Permissions(
        allow_http=True,
        allow_secrets=True,
    ),
    "slack": Permissions(
        allow_http=True,
        http_allowed_hosts=["slack.com", "api.slack.com"],
        allow_secrets=True,
        secret_allowed_names=["slack_oauth_token", "slack_bot_token"],
    ),
    "discord": Permissions(
        allow_http=True,
        http_allowed_hosts=["discord.com", "discordapp.com"],
        allow_secrets=True,
        secret_allowed_names=["discord_token", "discord_webhook_url"],
    ),
    "github": Permissions(
        allow_http=True,
        http_allowed_hosts=["api.github.com", "github.com"],
        allow_secrets=True,
        secret_allowed_names=["github_token", "github_app_key"],
    ),
    "google-drive": Permissions(
        allow_http=True,
        http_allowed_hosts=["googleapis.com", "www.googleapis.com"],
        allow_secrets=True,
        secret_allowed_names=["google_oauth_token"],
    ),
}
