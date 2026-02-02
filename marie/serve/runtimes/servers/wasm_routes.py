"""
Wasm Compilation Routes for Marie Gateway.

This module provides REST API endpoints for compiling user code
to WebAssembly components.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Response

logger = logging.getLogger(__name__)

# Lazy-loaded compiler service
_wasm_compiler = None


def _get_compiler():
    """Get or create the Wasm compiler service."""
    global _wasm_compiler
    if _wasm_compiler is None:
        try:
            from marie_wasm import WasmCompilerService

            _wasm_compiler = WasmCompilerService()
        except ImportError:
            logger.warning("marie-wasm package not installed")
            return None
    return _wasm_compiler


def register_wasm_routes(app: FastAPI) -> None:
    """
    Register Wasm compilation routes on the FastAPI app.

    Args:
        app: FastAPI application instance
    """

    @app.post(
        path="/api/nodes/{node_id}/compile",
        summary="Compile code node to Wasm",
        tags=["Wasm Compilation"],
    )
    async def compile_node(
        node_id: str,
        code: str,
        language: str,
        dependencies: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 120,
        response: Response = None,
    ):
        """
        Compile user code to a WebAssembly component.

        Called by marie-studio when user saves a Code node.

        Args:
            node_id: Unique identifier for the node
            code: Source code to compile
            language: Programming language (rust, python, js)
            dependencies: Optional dict of additional files
            timeout_seconds: Compilation timeout

        Returns:
            Compiled module path or error
        """
        try:
            from marie_wasm import Language

            compiler = _get_compiler()
            if compiler is None:
                response.status_code = 501
                return {
                    "success": False,
                    "error": "marie-wasm package not installed",
                }

            # Parse language
            try:
                lang = Language.from_string(language)
            except ValueError:
                response.status_code = 400
                return {
                    "success": False,
                    "error": f"Unsupported language: {language}. "
                    f"Supported: rust, python, js",
                }

            # Compile
            wasm_path = await compiler.compile(
                code=code,
                language=lang,
                node_id=node_id,
                dependencies=dependencies,
                timeout_seconds=timeout_seconds,
            )

            return {
                "success": True,
                "wasm_path": wasm_path,
                "node_id": node_id,
            }

        except ImportError:
            response.status_code = 501
            return {
                "success": False,
                "error": "marie-wasm package not installed",
            }
        except Exception as e:
            logger.error(f"Wasm compilation failed: {e}")
            response.status_code = 500
            return {
                "success": False,
                "error": str(e),
            }

    @app.get(
        path="/api/wasm/languages",
        summary="List supported Wasm compilation languages",
        tags=["Wasm Compilation"],
    )
    async def list_wasm_languages():
        """List supported programming languages for Wasm compilation."""
        try:
            from marie_wasm import Language

            return {
                "languages": [lang.value for lang in Language],
            }
        except ImportError:
            return {
                "languages": [],
                "error": "marie-wasm package not installed",
            }

    @app.get(
        path="/api/wasm/status",
        summary="Get Wasm compiler status",
        tags=["Wasm Compilation"],
    )
    async def get_wasm_status():
        """Get status of the Wasm compiler service."""
        try:
            compiler = _get_compiler()
            if compiler is None:
                return {
                    "available": False,
                    "error": "marie-wasm package not installed",
                }

            return {
                "available": compiler.is_available(),
                "languages": [
                    lang.value for lang in compiler.get_supported_languages()
                ],
            }
        except ImportError:
            return {
                "available": False,
                "error": "marie-wasm package not installed",
            }
