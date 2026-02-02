"""
marie-wasm: Wasmtime-based runtime for Marie workflow nodes.

This package provides compilation and host function implementations
for executing workflow nodes in WebAssembly.
"""

from marie_wasm.compiler import CompilationError, WasmCompilerService
from marie_wasm.host import HostImplementations
from marie_wasm.types import (
    CompilerConfig,
    DataItem,
    ExecutionContext,
    ExecutionResult,
    Language,
    Permissions,
)

__version__ = "0.1.0"

__all__ = [
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
