"""
marie-wasm: Wasmtime-based runtime for Marie workflow nodes.

This package provides compilation and host function implementations
for executing workflow nodes in WebAssembly.
"""

from pathlib import Path

from marie.wasm.compiler import CompilationError, WasmCompilerService
from marie.wasm.host import HostImplementations
from marie.wasm.types import (
    CompilerConfig,
    DataItem,
    ExecutionContext,
    ExecutionResult,
    Language,
    Permissions,
)

__version__ = "0.1.0"

# The built-in WASM node library ships with this package. `__file__` is
# .../packages/marie-wasm/src/marie/wasm/__init__.py, so parents[3] is the
# package root (.../packages/marie-wasm). Use these instead of CWD-relative
# paths. (For non-editable wheels the `nodes/` tree must be included as package
# data — tracked as a packaging follow-up.)
NODES_DIR = Path(__file__).resolve().parents[3] / "nodes"
BUILTIN_NODES_DIR = NODES_DIR / "compiled"

__all__ = [
    # Built-in node library paths
    "NODES_DIR",
    "BUILTIN_NODES_DIR",
    # Types
    "Language",
    "Permissions",
    "ExecutionContext",
    "DataItem",
    "ExecutionResult",
    "CompilerConfig",
    # Compiler
    "WasmCompilerService",
    "CompilationError",
    # Host
    "HostImplementations",
]
