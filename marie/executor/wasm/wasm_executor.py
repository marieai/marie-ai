"""
Wasm Node Executor.

Executes pre-compiled WebAssembly components using Wasmtime.
Components are loaded from storage (S3/PostgreSQL) or local disk.

Isolation model:
- Wasmtime sandbox: primary security boundary (memory, capability, fuel)
- ProcessPoolExecutor: blast-radius containment if wasmtime has a bug.
  A SIGSEGV in a worker kills that process only; the pool replaces it
  and the executor survives.
"""

import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Optional

from docarray import BaseDoc, DocList
from docarray.documents import TextDoc

from marie import Executor, requests
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger

# Optional marie_wasm import (main process only — for permission resolution)
try:
    from marie_wasm import BUILTIN_PERMISSIONS, Permissions

    MARIE_WASM_AVAILABLE = True
except ImportError:
    MARIE_WASM_AVAILABLE = False

# Check wasmtime at import time so status endpoint can report it
try:
    import wasmtime  # noqa: F401

    WASMTIME_AVAILABLE = True
except ImportError:
    WASMTIME_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# Worker-process globals.  Initialised once per pool worker via
# _init_worker(); never touched by the main executor process.
# ──────────────────────────────────────────────────────────────────────
_worker_engine: Any = None
_worker_epoch_stop: Optional[threading.Event] = None
_worker_epoch_thread: Optional[threading.Thread] = None
_worker_component_cache: Optional["LRUCache"] = None
_worker_http_client: Any = None


def _init_worker(
    max_cached_components: int = 100,
    http_timeout: float = 30.0,
) -> None:
    """Called once when a pool worker process starts.

    Creates the wasmtime Engine and epoch ticker thread so that the
    first real execution doesn't pay the cold-start cost.
    """
    global _worker_engine, _worker_epoch_stop, _worker_epoch_thread
    global _worker_component_cache, _worker_http_client

    from wasmtime import Config, Engine

    config = Config()
    config.consume_fuel = True
    config.epoch_interruption = True
    config.wasm_component_model = True
    _worker_engine = Engine(config)
    _worker_component_cache = LRUCache(max_cached_components)

    # Epoch ticker — 10 ms granularity, same as the old in-process thread
    _worker_epoch_stop = threading.Event()

    def _tick() -> None:
        while not _worker_epoch_stop.wait(timeout=0.01):
            _worker_engine.increment_epoch()

    _worker_epoch_thread = threading.Thread(target=_tick, daemon=True)
    _worker_epoch_thread.start()

    # Lazy HTTP client — created on first use inside the worker
    try:
        import httpx

        _worker_http_client = httpx.Client(timeout=http_timeout)
    except ImportError:
        _worker_http_client = None

    logging.getLogger("wasm_worker").debug("WASM worker %d initialised", os.getpid())


def _execute_in_worker(
    wasm_bytes: bytes,
    input_data: list[dict],
    config: dict,
    context: dict,
    permissions_dict: dict,
    credentials: Optional[dict[str, str]],
    cache_key: str,
) -> dict:
    """Run a single WASM execution inside a pool worker process.

    Everything here is process-isolated from the main executor.
    A wasmtime Cranelift JIT bug that causes a SIGSEGV kills only
    this worker; the ProcessPoolExecutor replaces it transparently.

    All arguments are plain dicts/bytes — no unpicklable wasmtime objects
    cross the process boundary.
    """
    from wasmtime import Component, Linker, Store

    if _worker_engine is None:
        return {"success": False, "error": "Worker not initialised"}

    # ── Compile or fetch from worker-local cache ──────────────────
    component = _worker_component_cache.get(cache_key)
    if component is None:
        try:
            component = Component(_worker_engine, wasm_bytes)
            _worker_component_cache.put(cache_key, component)
        except Exception as e:
            return {"success": False, "error": f"Component compile error: {e}"}

    # ── Reconstruct permissions in worker ─────────────────────────
    try:
        from marie_wasm import HostImplementations, Permissions

        permissions = Permissions(
            allow_http=permissions_dict.get("allow_http", False),
            http_allowed_hosts=permissions_dict.get("http_allowed_hosts", []),
            allow_secrets=permissions_dict.get("allow_secrets", False),
            secret_allowed_names=permissions_dict.get("secret_allowed_names", []),
            allow_kv=permissions_dict.get("allow_kv", False),
            kv_prefix=permissions_dict.get("kv_prefix", ""),
            max_memory_mb=permissions_dict.get("max_memory_mb", 64),
            max_fuel=permissions_dict.get("max_fuel", 1_000_000_000),
            timeout_ms=permissions_dict.get("timeout_ms", 30_000),
        )
        host = HostImplementations(
            permissions=permissions,
            credentials=credentials,
            http_client=_worker_http_client,
            execution_id=context.get("execution_id", ""),
        )
        marie_wasm_ok = True
    except ImportError:
        permissions = None
        host = None
        marie_wasm_ok = False

    # ── Create store with resource limits ─────────────────────────
    max_fuel = permissions_dict.get("max_fuel", 1_000_000_000)
    timeout_ms = permissions_dict.get("timeout_ms", 30_000)

    store = Store(_worker_engine)
    store.set_fuel(max_fuel)
    store.set_epoch_deadline(int(timeout_ms / 10))

    # ── Bind host functions ───────────────────────────────────────
    linker = Linker(_worker_engine)
    if host and marie_wasm_ok:
        bindings = host.get_bindings()
        for interface_name, functions in bindings.items():
            for func_name, func in functions.items():
                try:
                    linker.define_func(interface_name, func_name, func)
                except Exception:
                    pass  # non-fatal: component may not import this interface

    # ── Instantiate and execute ───────────────────────────────────
    try:
        instance = linker.instantiate(store, component)
        execute_func = instance.exports(store).get("execute")
        if execute_func is None:
            return {
                "success": False,
                "error": "Component does not export 'execute' function",
            }

        config_arg = {"json": json.dumps(config)}
        context_arg = {
            "workflow-id": context.get("workflow_id", ""),
            "execution-id": context.get("execution_id", ""),
            "node-id": context.get("node_id", ""),
            "run-index": context.get("run_index", 0),
        }

        start = time.monotonic()
        result = execute_func(store, input_data, config_arg, context_arg)
        elapsed = time.monotonic() - start

        fuel_remaining = store.get_fuel()
        fuel_consumed = max_fuel - fuel_remaining

        if "success" in result:
            return {
                "success": True,
                "data": result["success"],
                "fuel_consumed": fuel_consumed,
                "execution_time_ms": elapsed * 1000,
            }
        elif "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "fuel_consumed": fuel_consumed,
                "execution_time_ms": elapsed * 1000,
            }
        else:
            return {
                "success": False,
                "error": "Unexpected result format",
                "fuel_consumed": fuel_consumed,
            }

    except Exception as e:
        msg = str(e)
        if "epoch deadline" in msg.lower():
            return {"success": False, "error": "Execution timeout"}
        if "fuel" in msg.lower():
            return {"success": False, "error": "CPU limit exceeded"}
        return {"success": False, "error": f"Execution error: {msg}"}


# ──────────────────────────────────────────────────────────────────────
# Main-process classes
# ──────────────────────────────────────────────────────────────────────


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

    node_type: Optional[str] = None
    wasm_path: Optional[str] = None

    workflow_id: str = ""
    execution_id: str = ""
    node_id: str = ""
    run_index: int = 0

    config: str = "{}"
    permissions: Optional[dict] = None
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

    Isolation model:
    - Wasmtime sandbox: primary boundary (memory, capability, fuel)
    - ProcessPoolExecutor: blast-radius containment.  A wasmtime bug
      that crashes a worker kills that process only; the pool replaces
      it and the executor survives.

    Features:
    - Wasmtime component model API
    - Epoch-based timeout enforcement (per-worker epoch thread)
    - Fuel metering for CPU limits
    - LRU wasm-bytes cache (main process) + compiled-component cache (workers)
    - Host function implementations (HTTP, secrets, logging)
    """

    def __init__(
        self,
        builtin_nodes_dir: Path = Path("nodes/compiled"),
        max_cached_components: int = 100,
        default_timeout_ms: int = 30000,
        default_max_fuel: int = 1_000_000_000,
        worker_count: int = 0,
        http_timeout: float = 30.0,
        storage: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Args:
            builtin_nodes_dir: Path to pre-compiled built-in nodes.
            max_cached_components: LRU cache size (both main + per-worker).
            default_timeout_ms: Default execution timeout in milliseconds.
            default_max_fuel: Default CPU fuel limit.
            worker_count: Pool size.  0 = number of CPUs (capped at 8).
            http_timeout: Timeout for HTTP requests made by host functions.
            storage: S3/PostgreSQL config for loading user-compiled nodes.
        """
        kwargs["storage"] = storage
        super().__init__(**kwargs)

        self.logger = MarieLogger(self.__class__.__name__).logger
        self.builtin_nodes_dir = Path(builtin_nodes_dir)
        self.default_timeout_ms = default_timeout_ms
        self.default_max_fuel = default_max_fuel
        self._max_cached_components = max_cached_components
        self._http_timeout = http_timeout

        # Wasm bytes cache — keyed by cache_key, holds raw bytes.
        # Workers compile these into Components in their own caches.
        self._bytes_cache: LRUCache = LRUCache(max_cached_components)

        # Pre-load built-in node bytes
        self._load_builtin_bytes()

        # Worker pool
        if not WASMTIME_AVAILABLE:
            self.logger.warning(
                "wasmtime not installed — Wasm execution will fail. "
                "Install with: pip install wasmtime"
            )
            self._pool: Optional[ProcessPoolExecutor] = None
        else:
            count = worker_count or min(os.cpu_count() or 4, 8)
            mp_ctx = get_context("spawn")
            self._pool = ProcessPoolExecutor(
                max_workers=count,
                mp_context=mp_ctx,
                initializer=_init_worker,
                initargs=(max_cached_components, http_timeout),
            )
            self.logger.info("WASM process pool started with %d workers", count)

        if not MARIE_WASM_AVAILABLE:
            self.logger.warning(
                "marie_wasm not installed — host functions unavailable. "
                "Install with: pip install marie-wasm"
            )

        self.logger.info(
            "WasmNodeExecutor initialised — " "wasmtime: %s, marie_wasm: %s",
            WASMTIME_AVAILABLE,
            MARIE_WASM_AVAILABLE,
        )

    # ── Lifecycle ─────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Shutdown the worker pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self.logger.info("WasmNodeExecutor shutdown")

    # ── Bytes loading (main process) ──────────────────────────────

    def _load_builtin_bytes(self) -> None:
        """Pre-load raw bytes for all built-in .wasm files."""
        if not self.builtin_nodes_dir.exists():
            self.logger.debug(
                "Built-in nodes directory not found: %s", self.builtin_nodes_dir
            )
            return

        loaded = 0
        for wasm_file in self.builtin_nodes_dir.glob("*.wasm"):
            cache_key = f"builtin:{wasm_file.stem}"
            self._bytes_cache.put(cache_key, wasm_file.read_bytes())
            loaded += 1

        if loaded:
            self.logger.info("Pre-loaded %d built-in Wasm node(s)", loaded)

    async def _get_wasm_bytes(
        self, wasm_path: str, node_type: Optional[str]
    ) -> tuple[Optional[bytes], str]:
        """Load raw .wasm bytes and return (bytes, cache_key)."""
        # Built-in node
        if node_type:
            cache_key = f"builtin:{node_type}"
            cached = self._bytes_cache.get(cache_key)
            if cached is not None:
                return cached, cache_key

            wasm_file = self.builtin_nodes_dir / f"{node_type}.wasm"
            if wasm_file.exists():
                data = wasm_file.read_bytes()
                self._bytes_cache.put(cache_key, data)
                return data, cache_key

        # User-compiled node
        cache_key = f"user:{wasm_path}"
        cached = self._bytes_cache.get(cache_key)
        if cached is not None:
            return cached, cache_key

        try:
            if hasattr(self, "store") and self.store:
                data = await self.store.download(wasm_path)
            else:
                p = Path(wasm_path)
                if not p.exists():
                    self.logger.error("Wasm file not found: %s", wasm_path)
                    return None, cache_key
                data = p.read_bytes()

            self._bytes_cache.put(cache_key, data)
            return data, cache_key
        except Exception as e:
            self.logger.error("Failed to load wasm from %s: %s", wasm_path, e)
            return None, cache_key

    # ── Permission resolution (main process) ──────────────────────

    def _resolve_permissions(
        self, node_type: Optional[str], permissions_dict: Optional[dict]
    ) -> dict:
        """Resolve permissions to a plain dict for the worker."""
        defaults = {
            "allow_http": False,
            "http_allowed_hosts": [],
            "allow_secrets": False,
            "secret_allowed_names": [],
            "allow_kv": False,
            "kv_prefix": "",
            "max_memory_mb": 64,
            "max_fuel": self.default_max_fuel,
            "timeout_ms": self.default_timeout_ms,
        }

        if MARIE_WASM_AVAILABLE and node_type and node_type in BUILTIN_PERMISSIONS:
            perms = BUILTIN_PERMISSIONS[node_type]
            return {
                "allow_http": perms.allow_http,
                "http_allowed_hosts": list(perms.http_allowed_hosts),
                "allow_secrets": perms.allow_secrets,
                "secret_allowed_names": list(perms.secret_allowed_names),
                "allow_kv": perms.allow_kv,
                "kv_prefix": perms.kv_prefix,
                "max_memory_mb": perms.max_memory_mb,
                "max_fuel": perms.max_fuel,
                "timeout_ms": perms.timeout_ms,
            }

        if permissions_dict:
            for key in defaults:
                if key in permissions_dict:
                    defaults[key] = permissions_dict[key]
            return defaults

        return defaults

    # ── Request handlers ──────────────────────────────────────────

    @requests(on="/execute")
    async def execute(
        self,
        docs: DocList[WasmInputDoc],
        parameters: Optional[dict] = None,
        **kwargs,
    ) -> DocList[WasmOutputDoc]:
        """Execute a Wasm node."""
        params = WasmParameters(**(parameters or {}))

        if not params.node_type and not params.wasm_path:
            return DocList[WasmOutputDoc](
                [
                    WasmOutputDoc(
                        success=False,
                        error="Either node_type or wasm_path must be provided",
                    )
                ]
            )

        if self._pool is None:
            return DocList[WasmOutputDoc](
                [WasmOutputDoc(success=False, error="Wasmtime not available")]
            )

        # Load bytes in main process (may hit S3/storage)
        wasm_bytes, cache_key = await self._get_wasm_bytes(
            params.wasm_path or "", params.node_type
        )
        if wasm_bytes is None:
            return DocList[WasmOutputDoc](
                [
                    WasmOutputDoc(
                        success=False,
                        error=f"Failed to load: {params.node_type or params.wasm_path}",
                    )
                ]
            )

        permissions_dict = self._resolve_permissions(
            params.node_type, params.permissions
        )

        input_data = [
            {
                "json": doc.json_data,
                "binary": list(doc.binary) if doc.binary else None,
            }
            for doc in docs
        ]

        try:
            config = json.loads(params.config) if params.config else {}
        except json.JSONDecodeError:
            config = {}

        context = {
            "workflow_id": params.workflow_id,
            "execution_id": params.execution_id,
            "node_id": params.node_id,
            "run_index": params.run_index,
        }

        # Dispatch to worker pool
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._pool,
                _execute_in_worker,
                wasm_bytes,
                input_data,
                config,
                context,
                permissions_dict,
                params.credentials,
                cache_key,
            )
        except BrokenExecutor:
            self.logger.error("WASM worker pool crashed — reinitialising")
            self._recreate_pool()
            return DocList[WasmOutputDoc](
                [
                    WasmOutputDoc(
                        success=False,
                        error="Execution failed: worker process crashed",
                    )
                ]
            )

        # Build output
        if result.get("success"):
            output_docs = []
            for item in result.get("data", []):
                output_docs.append(
                    WasmOutputDoc(
                        json_data=item.get("json", "{}"),
                        binary=(bytes(item["binary"]) if item.get("binary") else None),
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
        pool = self._pool
        info = {
            "executor": "WasmNodeExecutor",
            "wasmtime_available": WASMTIME_AVAILABLE,
            "marie_wasm_available": MARIE_WASM_AVAILABLE,
            "cached_wasm_bytes": len(self._bytes_cache),
            "pool_alive": pool is not None and not getattr(pool, "_broken", False),
            "pool_workers": pool._max_workers if pool else 0,
        }
        return DocList[TextDoc]([TextDoc(text=json.dumps(info))])

    @requests(on="/list-nodes")
    async def list_nodes(self, **kwargs) -> DocList[TextDoc]:
        """List available built-in nodes."""
        nodes = []
        if self.builtin_nodes_dir.exists():
            for wasm_file in self.builtin_nodes_dir.glob("*.wasm"):
                nodes.append(wasm_file.stem)
        return DocList[TextDoc]([TextDoc(text=json.dumps({"nodes": nodes}))])

    # ── Internal ──────────────────────────────────────────────────

    def _recreate_pool(self) -> None:
        """Replace a broken pool with a fresh one."""
        old = self._pool
        if old is not None:
            try:
                old.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        count = old._max_workers if old else min(os.cpu_count() or 4, 8)
        mp_ctx = get_context("spawn")
        self._pool = ProcessPoolExecutor(
            max_workers=count,
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=(self._max_cached_components, self._http_timeout),
        )
        self.logger.info("WASM process pool recreated with %d workers", count)
