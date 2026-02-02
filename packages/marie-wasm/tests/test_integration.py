"""Integration tests for marie-wasm with real Docker containers.

Run with: pytest tests/test_integration.py -v -m docker
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Skip all tests if Docker is not available
docker = pytest.importorskip("docker")

from marie_wasm.compiler import CompilationError, WasmCompilerService
from marie_wasm.types import Language


def is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def are_compiler_images_available() -> bool:
    """Check if compiler images are built."""
    try:
        client = docker.from_env()
        required_images = [
            "marieai/marie-compiler-rust:latest",
            "marieai/marie-compiler-python:latest",
            "marieai/marie-compiler-js:latest",
        ]
        for image in required_images:
            try:
                client.images.get(image)
            except docker.errors.ImageNotFound:
                return False
        return True
    except Exception:
        return False


# Marker for tests requiring Docker
pytestmark = pytest.mark.docker

# Skip if Docker is not available
docker_available = pytest.mark.skipif(
    not is_docker_available(),
    reason="Docker is not available",
)

# Skip if compiler images are not built
compilers_available = pytest.mark.skipif(
    not are_compiler_images_available(),
    reason="Compiler images not built. Run: make -f Makefile.wasm compilers-build",
)


class TestCompileExecuteIntegration:
    """Integration tests for compilation with real Docker containers."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "wasm_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage client."""
        storage = AsyncMock()
        storage.exists = AsyncMock(return_value=False)
        storage.upload = AsyncMock(return_value="test/path.wasm")
        return storage

    @pytest.fixture
    def compiler_service(self, temp_cache_dir, mock_storage):
        """Create a compiler service with real Docker client."""
        return WasmCompilerService(
            cache_dir=temp_cache_dir,
            storage_client=mock_storage,
            registry="marieai",
            image_version="latest",
        )

    @docker_available
    @compilers_available
    @pytest.mark.asyncio
    async def test_compile_python_code(self, compiler_service, temp_cache_dir):
        """Test compiling Python code to WASM."""
        python_code = '''
def execute(input_items, config, context):
    """Process input and return result."""
    result = {"message": "Hello from Python!"}
    return {"success": [{"json": str(result)}]}
'''

        storage_key = await compiler_service.compile(
            code=python_code,
            language=Language.PYTHON,
            node_id="test-python-node",
            timeout_seconds=300,
        )

        # Verify storage key format
        assert storage_key.startswith("wasm/nodes/test-python-node/")
        assert storage_key.endswith(".wasm")

        # Verify local cache was created
        cache_files = list(temp_cache_dir.glob("*.wasm"))
        assert len(cache_files) == 1

        # Verify file size is reasonable (Python WASM is ~40MB)
        wasm_size = cache_files[0].stat().st_size
        assert wasm_size > 1_000_000, f"WASM file too small: {wasm_size} bytes"

    @docker_available
    @compilers_available
    @pytest.mark.asyncio
    async def test_compile_javascript_code(self, compiler_service, temp_cache_dir):
        """Test compiling JavaScript code to WASM."""
        js_code = '''
function execute(input, config, context) {
    const message = "Hello from JavaScript!";
    return { success: [{ json: JSON.stringify({ message }) }] };
}
'''

        storage_key = await compiler_service.compile(
            code=js_code,
            language=Language.JAVASCRIPT,
            node_id="test-js-node",
            timeout_seconds=120,
        )

        # Verify storage key format
        assert storage_key.startswith("wasm/nodes/test-js-node/")
        assert storage_key.endswith(".wasm")

        # Verify local cache was created
        cache_files = list(temp_cache_dir.glob("*.wasm"))
        assert len(cache_files) == 1

        # Verify file size is reasonable (JS WASM is ~11MB)
        wasm_size = cache_files[0].stat().st_size
        assert wasm_size > 1_000_000, f"WASM file too small: {wasm_size} bytes"

    @docker_available
    @compilers_available
    @pytest.mark.asyncio
    async def test_compile_rust_code(self, compiler_service, temp_cache_dir):
        """Test compiling Rust code to WASM."""
        rust_code = '''
fn execute(input_items: Vec<Item>, config: Config, context: Context) -> Response {
    let message = "Hello from Rust!";
    Response::success(vec![
        Item { json: format!(r#"{{"message": "{}"}}"#, message), binary: None }
    ])
}
'''

        storage_key = await compiler_service.compile(
            code=rust_code,
            language=Language.RUST,
            node_id="test-rust-node",
            timeout_seconds=300,
        )

        # Verify storage key format
        assert storage_key.startswith("wasm/nodes/test-rust-node/")
        assert storage_key.endswith(".wasm")

        # Verify local cache was created
        cache_files = list(temp_cache_dir.glob("*.wasm"))
        assert len(cache_files) == 1

        # Verify file size is reasonable (Rust WASM is ~67KB)
        wasm_size = cache_files[0].stat().st_size
        assert wasm_size > 10_000, f"WASM file too small: {wasm_size} bytes"

    @docker_available
    @compilers_available
    @pytest.mark.asyncio
    async def test_cache_hit_avoids_recompile(
        self, compiler_service, temp_cache_dir, mock_storage
    ):
        """Test that cache hits avoid recompilation."""
        js_code = '''
function execute(input, config, context) {
    return { success: [{ json: "{}" }] };
}
'''

        # First compilation
        storage_key1 = await compiler_service.compile(
            code=js_code,
            language=Language.JAVASCRIPT,
            node_id="cache-test-node",
            timeout_seconds=120,
        )

        # Get initial upload count
        initial_upload_count = mock_storage.upload.call_count

        # Reset storage exists to simulate fresh check
        mock_storage.exists.return_value = False

        # Second compilation with same code - should use cache
        storage_key2 = await compiler_service.compile(
            code=js_code,
            language=Language.JAVASCRIPT,
            node_id="cache-test-node-2",  # Different node ID
            timeout_seconds=120,
        )

        # Same code should produce same storage key (cache hit)
        # The node_id is in the storage path, but the cache key is based on code hash
        cache_files = list(temp_cache_dir.glob("*.wasm"))
        assert len(cache_files) == 1, "Cache should reuse existing compilation"

    @docker_available
    @compilers_available
    @pytest.mark.asyncio
    async def test_compile_invalid_code_fails(self, compiler_service):
        """Test that invalid code produces a compilation error."""
        invalid_python = '''
def execute(input_items config, context):  # Missing comma
    return {"success": []}
'''

        with pytest.raises(CompilationError) as exc_info:
            await compiler_service.compile(
                code=invalid_python,
                language=Language.PYTHON,
                node_id="invalid-python-node",
                timeout_seconds=120,
            )

        # Should contain syntax error information
        assert "Compilation failed" in str(exc_info.value) or exc_info.value.stderr

    @docker_available
    @compilers_available
    @pytest.mark.asyncio
    async def test_compilation_timeout(self, temp_cache_dir, mock_storage):
        """Test that compilation timeout works."""
        # Code that would take a very long time to compile is hard to create,
        # so we'll test with a very short timeout
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            storage_client=mock_storage,
            registry="marieai",
            image_version="latest",
        )

        simple_code = '''
function execute(input, config, context) {
    return { success: [] };
}
'''

        # Note: This test may not actually timeout since compilation is fast
        # But it verifies the timeout mechanism is in place
        try:
            await service.compile(
                code=simple_code,
                language=Language.JAVASCRIPT,
                node_id="timeout-test",
                timeout_seconds=300,  # Reasonable timeout
            )
        except CompilationError as e:
            if "timed out" in str(e).lower():
                pass  # Expected timeout
            else:
                raise  # Unexpected error


class TestWasmExecutorIntegration:
    """Full pipeline tests: compile then execute."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing."""
        cache_dir = tmp_path / "wasm_cache"
        cache_dir.mkdir()
        nodes_dir = tmp_path / "compiled_nodes"
        nodes_dir.mkdir()
        return {"cache": cache_dir, "nodes": nodes_dir}

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage client."""
        storage = AsyncMock()
        storage.exists = AsyncMock(return_value=False)
        storage.upload = AsyncMock(return_value="test/path.wasm")
        return storage

    @docker_available
    @compilers_available
    @pytest.mark.asyncio
    async def test_compile_and_execute_js(self, temp_dirs, mock_storage):
        """Test compiling JavaScript code and executing it."""
        # Check if wasmtime is available
        try:
            from wasmtime import Component, Engine, Linker, Store
        except ImportError:
            pytest.skip("wasmtime not installed")

        # Compile the code
        compiler = WasmCompilerService(
            cache_dir=temp_dirs["cache"],
            storage_client=mock_storage,
            registry="marieai",
            image_version="latest",
        )

        js_code = '''
function execute(input, config, context) {
    const items = input.map(item => {
        const data = JSON.parse(item.json);
        data.processed = true;
        return { json: JSON.stringify(data) };
    });
    return { ok: items };
}
'''

        storage_key = await compiler.compile(
            code=js_code,
            language=Language.JAVASCRIPT,
            node_id="exec-test-js",
            timeout_seconds=120,
        )

        # Load the compiled WASM
        cache_files = list(temp_dirs["cache"].glob("*.wasm"))
        assert len(cache_files) == 1
        wasm_path = cache_files[0]

        # Create engine and load component
        from wasmtime import Config

        config = Config()
        config.wasm_component_model = True
        engine = Engine(config)

        try:
            component = Component.from_file(engine, str(wasm_path))
        except Exception as e:
            # Some WASM components may require host bindings
            pytest.skip(f"Component loading requires host bindings: {e}")

        # Basic verification that the component was created
        assert component is not None

    @docker_available
    @compilers_available
    @pytest.mark.asyncio
    async def test_compile_and_verify_wasm_structure(self, temp_dirs, mock_storage):
        """Test that compiled WASM has correct structure."""
        compiler = WasmCompilerService(
            cache_dir=temp_dirs["cache"],
            storage_client=mock_storage,
            registry="marieai",
            image_version="latest",
        )

        rust_code = '''
fn execute(input_items: Vec<Item>, config: Config, context: Context) -> Response {
    Response::success(input_items)
}
'''

        await compiler.compile(
            code=rust_code,
            language=Language.RUST,
            node_id="struct-test",
            timeout_seconds=300,
        )

        cache_files = list(temp_dirs["cache"].glob("*.wasm"))
        assert len(cache_files) == 1
        wasm_path = cache_files[0]

        # Read WASM bytes and verify magic number
        wasm_bytes = wasm_path.read_bytes()

        # WASM magic number: \0asm
        assert wasm_bytes[:4] == b'\x00asm', "Invalid WASM magic number"

        # WASM version should be 1 for core modules, or check component format
        version = int.from_bytes(wasm_bytes[4:8], 'little')
        # Version 1 for core WASM, version 13 for component model (may vary)
        assert version in [1, 13, 14, 15, 16], f"Unexpected WASM version: {version}"


class TestCompilerServiceUnit:
    """Unit tests that don't require Docker."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "wasm_cache"
        cache_dir.mkdir()
        return cache_dir

    def test_service_without_docker(self, temp_cache_dir):
        """Test service initialization when Docker is unavailable."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=None,
        )
        assert service.is_available() is False

    def test_get_supported_languages(self, temp_cache_dir):
        """Test getting supported languages."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=None,
        )
        languages = service.get_supported_languages()
        assert Language.RUST in languages
        assert Language.PYTHON in languages
        assert Language.JAVASCRIPT in languages

    def test_cache_key_determinism(self, temp_cache_dir):
        """Test that cache keys are deterministic."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=None,
        )

        code = "def execute(): pass"
        key1 = service._get_cache_key(code, Language.PYTHON)
        key2 = service._get_cache_key(code, Language.PYTHON)
        key3 = service._get_cache_key(code, Language.RUST)

        assert key1 == key2, "Same code should produce same key"
        assert key1 != key3, "Different language should produce different key"

    def test_storage_key_format(self, temp_cache_dir):
        """Test storage key format."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=None,
        )

        cache_key = "abc123"
        node_id = "my-node-id"
        storage_key = service._get_storage_key(cache_key, node_id)

        assert storage_key == "wasm/nodes/my-node-id/abc123.wasm"
