"""
Wasm Compiler Service.

Docker-based compilation of user code to WebAssembly components.
Called during node build from the UI (via Gateway).
"""

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Protocol

try:
    import docker
    from docker.errors import ContainerError, ImageNotFound

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None  # type: ignore

from marie_wasm.types import (
    COMPILER_CONFIGS,
    CompileRequest,
    CompileResponse,
    Language,
)

logger = logging.getLogger(__name__)


class CompilationError(Exception):
    """Raised when code compilation fails."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


class StorageClient(Protocol):
    """Protocol for storage backend."""

    async def upload(self, key: str, data: bytes) -> str:
        """Upload data and return the storage path."""
        ...

    async def download(self, key: str) -> bytes:
        """Download data by key."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...


class WasmCompilerService:
    """
    Docker-based Wasm compiler called during node build from UI.

    This service:
    1. Wraps user code with language-specific templates
    2. Compiles wrapped code in Docker containers
    3. Stores compiled .wasm modules in storage (S3/PostgreSQL)
    """

    def __init__(
        self,
        storage_client: Optional[StorageClient] = None,
        template_dir: Optional[Path] = None,
        cache_dir: Path = Path("/var/cache/marie/wasm"),
        docker_client: Optional[Any] = None,
        image_version: str = "latest",
        registry: str = "marieai",
        max_cache_size: int = 1000,
    ):
        """
        Initialize the compiler service.

        Args:
            storage_client: Storage backend for compiled modules
            template_dir: Directory containing code wrapper templates
            cache_dir: Local cache directory for compiled modules
            docker_client: Docker client instance (created if not provided)
            image_version: Version tag for compiler images
            registry: Docker registry for compiler images
            max_cache_size: Maximum number of cached modules
        """
        self.storage = storage_client
        self.template_dir = template_dir or self._default_template_dir()
        self.cache_dir = cache_dir
        self.image_version = image_version
        self.registry = registry
        self.max_cache_size = max_cache_size

        # Initialize Docker client
        if docker_client:
            self.docker = docker_client
        elif DOCKER_AVAILABLE:
            self.docker = docker.from_env()
        else:
            self.docker = None
            logger.warning("Docker not available - compilation will fail")

        # Load templates
        self.templates = self._load_templates()

        # Compilation locks to prevent duplicate compilations
        self._compile_locks: dict[str, asyncio.Lock] = {}

        # LRU cache tracking
        self._cache_order: list[str] = []

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _default_template_dir(self) -> Path:
        """Get default template directory relative to this module."""
        return Path(__file__).parent / "templates"

    def _load_templates(self) -> dict[Language, str]:
        """Load code wrapper templates for each language."""
        templates = {}
        template_files = {
            Language.PYTHON: "python.py.tmpl",
            Language.JAVASCRIPT: "javascript.js.tmpl",
            Language.RUST: "rust.rs.tmpl",
        }

        for lang, filename in template_files.items():
            template_path = self.template_dir / filename
            if template_path.exists():
                templates[lang] = template_path.read_text()
            else:
                logger.warning(f"Template not found: {template_path}")

        return templates

    def _wrap_user_code(self, code: str, language: Language) -> str:
        """
        Wrap user code with language-specific template.

        The template provides:
        - Error handling with structured responses
        - Input/config/context injection
        - Return type validation
        - Logging integration
        """
        template = self.templates.get(language)
        if not template:
            raise CompilationError(f"No template available for language: {language}")

        return template.replace("{USER_CODE}", code)

    def _get_image_name(self, language: Language) -> str:
        """Get Docker image name for a language compiler."""
        config = COMPILER_CONFIGS.get(language)
        suffix = config.image_suffix if config else language.value
        return f"{self.registry}/marie-compiler-{suffix}:{self.image_version}"

    def _get_cache_key(self, code: str, language: Language, deps_hash: str = "") -> str:
        """Generate cache key from code hash and dependencies."""
        content = f"{language.value}:{deps_hash}:{code}"
        return hashlib.sha256(content.encode()).hexdigest()[:24]

    def _get_cached_path(self, cache_key: str) -> Path:
        """Get path to locally cached Wasm component."""
        return self.cache_dir / f"{cache_key}.wasm"

    def _get_storage_key(self, cache_key: str, node_id: str) -> str:
        """Get storage key for a compiled module."""
        return f"wasm/nodes/{node_id}/{cache_key}.wasm"

    def _evict_cache_if_needed(self) -> None:
        """Evict oldest cache entries if over limit."""
        while len(self._cache_order) > self.max_cache_size:
            oldest_key = self._cache_order.pop(0)
            cached_path = self._get_cached_path(oldest_key)
            if cached_path.exists():
                cached_path.unlink()
                logger.debug(f"Evicted cache entry: {oldest_key}")

    def _update_cache_lru(self, cache_key: str) -> None:
        """Update LRU order for cache key."""
        if cache_key in self._cache_order:
            self._cache_order.remove(cache_key)
        self._cache_order.append(cache_key)

    async def compile(
        self,
        code: str,
        language: Language,
        node_id: str,
        dependencies: Optional[dict[str, str]] = None,
        timeout_seconds: int = 120,
    ) -> str:
        """
        Compile source code to a Wasm component and store in S3.

        Args:
            code: Source code to compile
            language: Programming language
            node_id: Unique identifier for the node
            dependencies: Optional dict of additional files (filename -> content)
            timeout_seconds: Compilation timeout

        Returns:
            Storage path to compiled .wasm module
        """
        start_time = time.monotonic()

        # Calculate dependency hash
        deps_hash = ""
        if dependencies:
            deps_content = "".join(f"{k}:{v}" for k, v in sorted(dependencies.items()))
            deps_hash = hashlib.sha256(deps_content.encode()).hexdigest()[:8]

        cache_key = self._get_cache_key(code, language, deps_hash)
        cached_path = self._get_cached_path(cache_key)
        storage_key = self._get_storage_key(cache_key, node_id)

        # Check if already in storage
        if self.storage:
            try:
                if await self.storage.exists(storage_key):
                    logger.info(f"Module already compiled: {storage_key}")
                    return storage_key
            except Exception as e:
                logger.warning(f"Storage check failed: {e}")

        # Check local cache
        if cached_path.exists():
            self._update_cache_lru(cache_key)
            logger.info(f"Using cached module: {cached_path}")

            # Upload to storage if available
            if self.storage:
                wasm_bytes = cached_path.read_bytes()
                await self.storage.upload(storage_key, wasm_bytes)

            return storage_key

        # Ensure only one compilation per cache key
        if cache_key not in self._compile_locks:
            self._compile_locks[cache_key] = asyncio.Lock()

        async with self._compile_locks[cache_key]:
            # Double-check cache after acquiring lock
            if cached_path.exists():
                self._update_cache_lru(cache_key)
                if self.storage:
                    wasm_bytes = cached_path.read_bytes()
                    await self.storage.upload(storage_key, wasm_bytes)
                return storage_key

            # Wrap user code with template
            wrapped_code = self._wrap_user_code(code, language)

            # Compile in Docker container
            wasm_bytes = await self._compile_in_container(
                wrapped_code, language, dependencies, timeout_seconds
            )

            # Cache locally
            self._evict_cache_if_needed()
            cached_path.write_bytes(wasm_bytes)
            self._update_cache_lru(cache_key)

            # Upload to storage
            if self.storage:
                await self.storage.upload(storage_key, wasm_bytes)

            compile_time = time.monotonic() - start_time
            logger.info(
                f"Compiled {language.value} module: {len(wasm_bytes)} bytes "
                f"in {compile_time:.2f}s"
            )

            return storage_key

    async def compile_request(self, request: CompileRequest) -> CompileResponse:
        """
        Compile code from a CompileRequest.

        Convenience method for Gateway integration.
        """
        start_time = time.monotonic()

        try:
            wasm_path = await self.compile(
                code=request.code,
                language=request.language,
                node_id=request.node_id,
                dependencies=request.dependencies,
                timeout_seconds=request.timeout_seconds,
            )

            compile_time_ms = (time.monotonic() - start_time) * 1000

            # Get size from cache
            cache_key = self._get_cache_key(
                request.code,
                request.language,
                (
                    hashlib.sha256(
                        "".join(
                            f"{k}:{v}"
                            for k, v in sorted((request.dependencies or {}).items())
                        ).encode()
                    ).hexdigest()[:8]
                    if request.dependencies
                    else ""
                ),
            )
            cached_path = self._get_cached_path(cache_key)
            wasm_size = cached_path.stat().st_size if cached_path.exists() else 0

            return CompileResponse.ok(
                wasm_path=wasm_path,
                compile_time_ms=compile_time_ms,
                wasm_size_bytes=wasm_size,
            )

        except CompilationError as e:
            logger.error(f"Compilation failed: {e}")
            return CompileResponse.err(str(e))
        except Exception as e:
            logger.exception(f"Unexpected compilation error: {e}")
            return CompileResponse.err(f"Internal error: {e}")

    async def _compile_in_container(
        self,
        code: str,
        language: Language,
        dependencies: Optional[dict[str, str]],
        timeout_seconds: int,
    ) -> bytes:
        """Run compilation in Docker container with security hardening."""
        if not self.docker:
            raise CompilationError("Docker not available for compilation")

        image = self._get_image_name(language)
        config = COMPILER_CONFIGS.get(language)

        if not config:
            raise CompilationError(f"No compiler config for language: {language}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write source code
            source_file = tmpdir_path / config.source_filename
            source_file.write_text(code)

            # Write additional dependency files if provided
            if dependencies and config.additional_files:
                for filename, content in dependencies.items():
                    if filename in config.additional_files:
                        (tmpdir_path / filename).write_text(content)

            # Run compiler container
            loop = asyncio.get_event_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: self._run_container(image, tmpdir_path)
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise CompilationError(
                    f"Compilation timed out after {timeout_seconds}s"
                )

            # Read compiled output
            output_file = tmpdir_path / "output.wasm"
            if not output_file.exists():
                raise CompilationError("Compilation produced no output")

            return output_file.read_bytes()

    def _run_container(self, image: str, workspace: Path) -> None:
        """Run Docker container for compilation with security hardening."""
        try:
            result = self.docker.containers.run(
                image,
                volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
                # Security hardening
                user=f"{os.getuid()}:{os.getgid()}",
                read_only=True,
                tmpfs={"/tmp": "rw,noexec,nosuid,size=256m"},
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"],
                network_disabled=True,
                # Resource limits
                mem_limit="1g",
                memswap_limit="1g",
                cpu_period=100000,
                cpu_quota=100000,  # 100% of one CPU
                pids_limit=256,
                # Execution
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
            )
            logger.debug(f"Container output: {result.decode() if result else ''}")

        except ContainerError as e:
            stderr = e.stderr.decode("utf-8") if e.stderr else ""
            logger.error(f"Container error: {stderr}")
            raise CompilationError(f"Compilation failed: {e}", stderr=stderr)

        except ImageNotFound:
            raise CompilationError(
                f"Compiler image not found: {image}. "
                f"Run 'make compilers-build' to build compiler containers."
            )

        except Exception as e:
            raise CompilationError(f"Docker error: {e}")

    def get_supported_languages(self) -> list[Language]:
        """Get list of supported languages."""
        return list(Language)

    def is_available(self) -> bool:
        """Check if compiler service is available."""
        return self.docker is not None and DOCKER_AVAILABLE
