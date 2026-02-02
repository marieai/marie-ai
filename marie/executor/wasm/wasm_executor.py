"""
Wasm Node Executor.

Executes pre-compiled WebAssembly components using Wasmtime.
Components are loaded from storage (S3/PostgreSQL) or local disk.
"""

import asyncio
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from docarray import BaseDoc, DocList
from docarray.documents import TextDoc

from marie import Executor, requests
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger

# Optional wasmtime import
try:
    from wasmtime import Component, Config, Engine, Linker, Store

    WASMTIME_AVAILABLE = True
except ImportError:
    WASMTIME_AVAILABLE = False
    Config = None  # type: ignore
    Engine = None  # type: ignore
    Store = None  # type: ignore
    Component = None  # type: ignore
    Linker = None  # type: ignore

# Optional marie_wasm import
try:
    from marie_wasm import (
        BUILTIN_PERMISSIONS,
        DataItem,
        ExecutionContext,
        ExecutionResult,
        HostImplementations,
        Permissions,
    )

    MARIE_WASM_AVAILABLE = True
except ImportError:
    MARIE_WASM_AVAILABLE = False


class WasmInputDoc(BaseDoc):
    """Input document for Wasm execution."""

    json_data: str = "{}"
    binary: Optional[bytes] = None


class WasmOutputDoc(BaseDoc):
    """Output document from Wasm execution."""

    json_data: str = "{}"
    binary: Optional[bytes] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class WasmParameters:
    """Parameters for Wasm execution."""

    # Either node_type (built-in) or wasm_path (user code)
    node_type: Optional[str] = None
    wasm_path: Optional[str] = None

    # Execution context
    workflow_id: str = ""
    execution_id: str = ""
    node_id: str = ""
    run_index: int = 0

    # Node configuration (JSON)
    config: str = "{}"

    # Permissions (for user code)
    permissions: Optional[dict] = None

    # Credentials for secrets interface
    credentials: Optional[dict[str, str]] = None


class LRUCache(OrderedDict):
    """Simple LRU cache with max size."""

    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

    def put(self, key: Any, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self.maxsize:
            self.popitem(last=False)


class WasmNodeExecutor(MarieExecutor):
    """
    Marie executor that runs pre-compiled Wasm components.

    Loads .wasm modules from storage (compiled at build time by Gateway)
    and executes them with the Wasmtime runtime.

    Features:
    - Wasmtime component model API
    - Epoch-based timeout enforcement
    - Fuel metering for CPU limits
    - LRU component cache
    - Host function implementations (HTTP, secrets, logging)
    """

    def __init__(
        self,
        builtin_nodes_dir: Path = Path("nodes/compiled"),
        max_cached_components: int = 100,
        default_timeout_ms: int = 30000,
        default_max_fuel: int = 1_000_000_000,
        storage: Optional[dict[str, Any]] = None,
        http_client: Optional[Callable] = None,
        kv_store: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the Wasm node executor.

        Args:
            builtin_nodes_dir: Path to pre-compiled built-in nodes
            max_cached_components: Maximum number of cached components
            default_timeout_ms: Default execution timeout in milliseconds
            default_max_fuel: Default CPU fuel limit
            storage: S3/PostgreSQL config for loading user-compiled nodes
            http_client: HTTP client for host functions
            kv_store: Key-value store for host functions
            **kwargs: Additional MarieExecutor arguments
        """
        kwargs["storage"] = storage
        super().__init__(**kwargs)

        self.logger = MarieLogger(self.__class__.__name__).logger
        self.builtin_nodes_dir = Path(builtin_nodes_dir)
        self.default_timeout_ms = default_timeout_ms
        self.default_max_fuel = default_max_fuel
        self._http_client = http_client
        self._kv_store = kv_store

        # Check dependencies
        if not WASMTIME_AVAILABLE:
            self.logger.warning(
                "wasmtime not installed - Wasm execution will fail. "
                "Install with: pip install wasmtime"
            )
            self.engine = None
        else:
            # Configure Wasmtime engine for component model
            config = Config()
            config.consume_fuel = True
            config.epoch_interruption = True
            config.wasm_component_model = True
            self.engine = Engine(config)

        if not MARIE_WASM_AVAILABLE:
            self.logger.warning(
                "marie_wasm not installed - host functions unavailable. "
                "Install with: pip install marie-wasm"
            )

        # Component cache (built-in + user code)
        self._component_cache: LRUCache = LRUCache(max_cached_components)

        # Epoch increment thread for timeout enforcement
        self._epoch_thread: Optional[threading.Thread] = None
        self._epoch_stop = threading.Event()
        if self.engine:
            self._start_epoch_thread()

        # Load built-in nodes
        self._load_builtin_nodes()

        self.logger.info(
            f"WasmNodeExecutor initialized - "
            f"wasmtime: {WASMTIME_AVAILABLE}, "
            f"marie_wasm: {MARIE_WASM_AVAILABLE}"
        )

    def _start_epoch_thread(self) -> None:
        """Start background thread that increments engine epoch for timeouts."""

        def epoch_incrementer() -> None:
            while not self._epoch_stop.wait(timeout=0.01):  # 10ms granularity
                if self.engine:
                    self.engine.increment_epoch()

        self._epoch_thread = threading.Thread(target=epoch_incrementer, daemon=True)
        self._epoch_thread.start()
        self.logger.debug("Epoch thread started")

    def shutdown(self) -> None:
        """Shutdown the executor and stop epoch thread."""
        self._epoch_stop.set()
        if self._epoch_thread:
            self._epoch_thread.join(timeout=1.0)
        self.logger.info("WasmNodeExecutor shutdown")

    def _load_builtin_nodes(self) -> None:
        """Pre-load all built-in node components from disk."""
        if not self.engine:
            return

        if not self.builtin_nodes_dir.exists():
            self.logger.debug(
                f"Built-in nodes directory not found: {self.builtin_nodes_dir}"
            )
            return

        loaded = 0
        for wasm_file in self.builtin_nodes_dir.glob("*.wasm"):
            node_type = wasm_file.stem
            try:
                component = Component.from_file(self.engine, str(wasm_file))
                self._component_cache.put(f"builtin:{node_type}", component)
                loaded += 1
            except Exception as e:
                self.logger.warning(f"Failed to load builtin node {node_type}: {e}")

        if loaded > 0:
            self.logger.info(f"Loaded {loaded} built-in Wasm nodes")

    async def _load_component(
        self, wasm_path: str, node_type: Optional[str] = None
    ) -> Optional[Any]:
        """
        Load a Wasm component from storage or cache.

        Args:
            wasm_path: Storage path to .wasm file
            node_type: Built-in node type name

        Returns:
            Wasmtime Component or None if not found
        """
        if not self.engine:
            return None

        # Check for built-in node
        if node_type:
            cache_key = f"builtin:{node_type}"
            component = self._component_cache.get(cache_key)
            if component:
                return component

            # Try loading from disk
            wasm_file = self.builtin_nodes_dir / f"{node_type}.wasm"
            if wasm_file.exists():
                try:
                    component = Component.from_file(self.engine, str(wasm_file))
                    self._component_cache.put(cache_key, component)
                    return component
                except Exception as e:
                    self.logger.error(f"Failed to load builtin node {node_type}: {e}")
                    return None

        # Load from storage
        cache_key = f"user:{wasm_path}"
        component = self._component_cache.get(cache_key)
        if component:
            return component

        # Download from storage
        try:
            # Use storage mixin if available
            if hasattr(self, "store") and self.store:
                wasm_bytes = await self.store.download(wasm_path)
            else:
                # Fall back to local file
                wasm_file = Path(wasm_path)
                if wasm_file.exists():
                    wasm_bytes = wasm_file.read_bytes()
                else:
                    self.logger.error(f"Wasm file not found: {wasm_path}")
                    return None

            component = Component(self.engine, wasm_bytes)
            self._component_cache.put(cache_key, component)
            return component

        except Exception as e:
            self.logger.error(f"Failed to load component from {wasm_path}: {e}")
            return None

    def _get_permissions(
        self, node_type: Optional[str], permissions_dict: Optional[dict]
    ) -> "Permissions":
        """Get permissions for execution."""
        if not MARIE_WASM_AVAILABLE:
            # Return a basic permissions object
            @dataclass
            class BasicPermissions:
                allow_http: bool = False
                allow_secrets: bool = False
                allow_kv: bool = False
                kv_prefix: str = ""
                max_memory_mb: int = 64
                max_fuel: int = 1_000_000_000
                timeout_ms: int = 30000

            return BasicPermissions()

        # Built-in node permissions
        if node_type and node_type in BUILTIN_PERMISSIONS:
            return BUILTIN_PERMISSIONS[node_type]

        # User-provided permissions
        if permissions_dict:
            return Permissions(
                allow_http=permissions_dict.get("allow_http", False),
                http_allowed_hosts=permissions_dict.get("http_allowed_hosts", []),
                allow_secrets=permissions_dict.get("allow_secrets", False),
                secret_allowed_names=permissions_dict.get("secret_allowed_names", []),
                allow_kv=permissions_dict.get("allow_kv", False),
                kv_prefix=permissions_dict.get("kv_prefix", ""),
                max_memory_mb=permissions_dict.get("max_memory_mb", 64),
                max_fuel=permissions_dict.get("max_fuel", self.default_max_fuel),
                timeout_ms=permissions_dict.get("timeout_ms", self.default_timeout_ms),
            )

        # Default minimal permissions
        return Permissions()

    async def _execute_component(
        self,
        component: Any,
        input_data: list[dict],
        config: dict,
        context: dict,
        permissions: Any,
        credentials: Optional[dict[str, str]] = None,
    ) -> dict:
        """
        Execute a Wasm component with the given permissions.

        Args:
            component: Wasmtime Component
            input_data: List of input data items
            config: Node configuration
            context: Execution context
            permissions: Permission settings
            credentials: Secret credentials

        Returns:
            Dict with success/error and data
        """
        if not self.engine:
            return {"success": False, "error": "Wasmtime not available"}

        # Create store with resource limits
        store = Store(self.engine)
        store.set_fuel(permissions.max_fuel)

        # Set epoch deadline for timeout (10ms per epoch)
        epoch_deadline = int(permissions.timeout_ms / 10)
        store.set_epoch_deadline(epoch_deadline)

        # Create host implementations
        if MARIE_WASM_AVAILABLE:
            host = HostImplementations(
                permissions=permissions,
                credentials=credentials,
                http_client=self._http_client,
                kv_store=self._kv_store,
                logger_func=lambda level, msg: self.logger.log(
                    self._log_level_to_int(level), msg
                ),
                execution_id=context.get("execution-id", ""),
            )
        else:
            host = None

        # Create linker and bind host functions
        linker = Linker(self.engine)
        if host:
            self._bind_host_functions(linker, host)

        try:
            # Instantiate component
            instance = linker.instantiate(store, component)

            # Get the execute export
            execute_func = instance.exports(store).get("execute")
            if execute_func is None:
                return {
                    "success": False,
                    "error": "Component does not export 'execute' function",
                }

            # Prepare arguments
            config_arg = {"json": json.dumps(config)}
            context_arg = {
                "workflow-id": context.get("workflow_id", ""),
                "execution-id": context.get("execution_id", ""),
                "node-id": context.get("node_id", ""),
                "run-index": context.get("run_index", 0),
            }

            # Execute in thread pool to not block async loop
            loop = asyncio.get_event_loop()
            start_time = time.monotonic()

            result = await loop.run_in_executor(
                None,
                lambda: execute_func(store, input_data, config_arg, context_arg),
            )

            execution_time = time.monotonic() - start_time

            # Calculate fuel consumed
            fuel_remaining = store.get_fuel()
            fuel_consumed = permissions.max_fuel - fuel_remaining

            self.logger.debug(
                f"Wasm execution completed: "
                f"time={execution_time:.3f}s, fuel={fuel_consumed}"
            )

            # Parse result
            if "success" in result:
                return {
                    "success": True,
                    "data": result["success"],
                    "fuel_consumed": fuel_consumed,
                    "execution_time_ms": execution_time * 1000,
                }
            elif "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "fuel_consumed": fuel_consumed,
                    "execution_time_ms": execution_time * 1000,
                }
            else:
                return {
                    "success": False,
                    "error": "Unexpected result format",
                    "fuel_consumed": fuel_consumed,
                }

        except Exception as e:
            error_msg = str(e)
            if "epoch deadline" in error_msg.lower():
                return {"success": False, "error": "Execution timeout"}
            if "fuel" in error_msg.lower():
                return {"success": False, "error": "CPU limit exceeded"}
            self.logger.error(f"Wasm execution error: {e}")
            return {"success": False, "error": f"Execution error: {error_msg}"}

    def _log_level_to_int(self, level: str) -> int:
        """Convert log level string to logging int."""
        import logging

        levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warn": logging.WARNING,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }
        return levels.get(level.lower(), logging.INFO)

    def _bind_host_functions(self, linker: Any, host: Any) -> None:
        """Bind host function implementations to the linker."""
        # Get all bindings from host
        bindings = host.get_bindings()

        for interface_name, functions in bindings.items():
            for func_name, func in functions.items():
                try:
                    linker.define_func(interface_name, func_name, func)
                except Exception as e:
                    self.logger.debug(
                        f"Could not bind {interface_name}::{func_name}: {e}"
                    )

    @requests(on="/execute")
    async def execute(
        self,
        docs: DocList[WasmInputDoc],
        parameters: Optional[dict] = None,
        **kwargs,
    ) -> DocList[WasmOutputDoc]:
        """
        Execute a Wasm node.

        Args:
            docs: Input documents
            parameters: Execution parameters including:
                - node_type: Built-in node name
                - wasm_path: S3 path to user-compiled .wasm
                - workflow_id, execution_id, node_id, run_index
                - config: Node configuration JSON
                - permissions: Permission dict for user code
                - credentials: Secret credentials

        Returns:
            Output documents
        """
        params = WasmParameters(**(parameters or {}))

        # Validate parameters
        if not params.node_type and not params.wasm_path:
            return DocList[WasmOutputDoc](
                [
                    WasmOutputDoc(
                        success=False,
                        error="Either node_type or wasm_path must be provided",
                    )
                ]
            )

        # Load component
        component = await self._load_component(
            wasm_path=params.wasm_path or "",
            node_type=params.node_type,
        )

        if not component:
            return DocList[WasmOutputDoc](
                [
                    WasmOutputDoc(
                        success=False,
                        error=f"Failed to load component: {params.node_type or params.wasm_path}",
                    )
                ]
            )

        # Get permissions
        permissions = self._get_permissions(params.node_type, params.permissions)

        # Prepare input data
        input_data = [
            {"json": doc.json_data, "binary": list(doc.binary) if doc.binary else None}
            for doc in docs
        ]

        # Parse config
        try:
            config = json.loads(params.config) if params.config else {}
        except json.JSONDecodeError:
            config = {}

        # Build context
        context = {
            "workflow_id": params.workflow_id,
            "execution_id": params.execution_id,
            "node_id": params.node_id,
            "run_index": params.run_index,
        }

        # Execute
        result = await self._execute_component(
            component=component,
            input_data=input_data,
            config=config,
            context=context,
            permissions=permissions,
            credentials=params.credentials,
        )

        # Build output
        if result.get("success"):
            output_docs = []
            for item in result.get("data", []):
                output_docs.append(
                    WasmOutputDoc(
                        json_data=item.get("json", "{}"),
                        binary=bytes(item["binary"]) if item.get("binary") else None,
                        success=True,
                    )
                )
            return (
                DocList[WasmOutputDoc](output_docs)
                if output_docs
                else DocList[WasmOutputDoc]([WasmOutputDoc(success=True)])
            )
        else:
            return DocList[WasmOutputDoc](
                [
                    WasmOutputDoc(
                        success=False,
                        error=result.get("error", "Unknown error"),
                    )
                ]
            )

    @requests(on="/status")
    async def status(self, **kwargs) -> DocList[TextDoc]:
        """Health check endpoint."""
        status_info = {
            "executor": "WasmNodeExecutor",
            "wasmtime_available": WASMTIME_AVAILABLE,
            "marie_wasm_available": MARIE_WASM_AVAILABLE,
            "cached_components": len(self._component_cache),
            "epoch_thread_alive": (
                self._epoch_thread.is_alive() if self._epoch_thread else False
            ),
        }
        return DocList[TextDoc]([TextDoc(text=json.dumps(status_info))])

    @requests(on="/list-nodes")
    async def list_nodes(self, **kwargs) -> DocList[TextDoc]:
        """List available built-in nodes."""
        nodes = []
        if self.builtin_nodes_dir.exists():
            for wasm_file in self.builtin_nodes_dir.glob("*.wasm"):
                nodes.append(wasm_file.stem)

        return DocList[TextDoc]([TextDoc(text=json.dumps({"nodes": nodes}))])
