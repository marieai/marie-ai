"""Tests for marie_wasm.compiler module."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from marie_wasm.compiler import (
    CompilationError,
    WasmCompilerService,
)
from marie_wasm.types import Language


class TestWasmCompilerService:
    """Tests for WasmCompilerService."""

    @pytest.fixture
    def mock_docker_client(self):
        """Create a mock Docker client."""
        client = Mock()
        client.containers = Mock()
        return client

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "wasm_cache"
        cache_dir.mkdir()
        return cache_dir

    def test_initialization(self, temp_cache_dir, mock_docker_client):
        """Test service initialization."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=mock_docker_client,
        )
        assert service.cache_dir == temp_cache_dir
        assert service.docker == mock_docker_client

    def test_get_supported_languages(self, temp_cache_dir, mock_docker_client):
        """Test getting supported languages."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=mock_docker_client,
        )
        languages = service.get_supported_languages()
        assert Language.RUST in languages
        assert Language.PYTHON in languages
        assert Language.JAVASCRIPT in languages

    def test_is_available(self, temp_cache_dir, mock_docker_client):
        """Test availability check."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=mock_docker_client,
        )
        assert service.is_available() is True

        # Test without Docker
        service_no_docker = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=None,
        )
        # When docker_client is None and docker module isn't available
        service_no_docker.docker = None
        assert service_no_docker.is_available() is False

    def test_get_image_name(self, temp_cache_dir, mock_docker_client):
        """Test Docker image name generation."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=mock_docker_client,
            registry="myregistry",
            image_version="v1.0",
        )
        assert (
            service._get_image_name(Language.RUST)
            == "myregistry/marie-compiler-rust:v1.0"
        )
        assert (
            service._get_image_name(Language.PYTHON)
            == "myregistry/marie-compiler-python:v1.0"
        )
        assert (
            service._get_image_name(Language.JAVASCRIPT)
            == "myregistry/marie-compiler-js:v1.0"
        )

    def test_get_cache_key(self, temp_cache_dir, mock_docker_client):
        """Test cache key generation."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=mock_docker_client,
        )
        code = "def execute(): pass"

        key1 = service._get_cache_key(code, Language.PYTHON)
        key2 = service._get_cache_key(code, Language.PYTHON)
        key3 = service._get_cache_key(code, Language.RUST)
        key4 = service._get_cache_key("different code", Language.PYTHON)

        # Same code and language should produce same key
        assert key1 == key2
        # Different language should produce different key
        assert key1 != key3
        # Different code should produce different key
        assert key1 != key4

    def test_wrap_user_code_python(self, temp_cache_dir, mock_docker_client):
        """Test Python code wrapping."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=mock_docker_client,
        )

        user_code = '''
def execute(input_items, config, context):
    return {"success": []}
'''
        wrapped = service._wrap_user_code(user_code, Language.PYTHON)

        # Check that user code is included
        assert "def execute(input_items, config, context):" in wrapped
        assert "return {\"success\": []}" in wrapped
        # Check that wrapper code is present
        assert "_marie_execute" in wrapped

    def test_wrap_user_code_javascript(self, temp_cache_dir, mock_docker_client):
        """Test JavaScript code wrapping."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=mock_docker_client,
        )

        user_code = '''
function execute(input, config, context) {
    return { success: [] };
}
'''
        wrapped = service._wrap_user_code(user_code, Language.JAVASCRIPT)

        # Check that user code is included
        assert "function execute(input, config, context)" in wrapped
        assert "return { success: [] };" in wrapped
        # Check that wrapper code is present
        assert "_marie_execute" in wrapped

    def test_cache_eviction(self, temp_cache_dir, mock_docker_client):
        """Test LRU cache eviction."""
        service = WasmCompilerService(
            cache_dir=temp_cache_dir,
            docker_client=mock_docker_client,
            max_cache_size=3,
        )

        # Simulate adding cache entries
        for i in range(5):
            key = f"key_{i}"
            service._cache_order.append(key)
            (temp_cache_dir / f"{key}.wasm").write_bytes(b"dummy")

        # Evict should remove oldest entries
        service._evict_cache_if_needed()

        assert len(service._cache_order) == 3
        # First two should be evicted
        assert "key_0" not in service._cache_order
        assert "key_1" not in service._cache_order
        # Last three should remain
        assert "key_2" in service._cache_order
        assert "key_3" in service._cache_order
        assert "key_4" in service._cache_order


class TestCompilationError:
    """Tests for CompilationError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = CompilationError("Compilation failed")
        assert str(error) == "Compilation failed"
        assert error.stderr == ""

    def test_error_with_stderr(self):
        """Test error with stderr output."""
        error = CompilationError("Compilation failed", stderr="error: syntax error")
        assert str(error) == "Compilation failed"
        assert error.stderr == "error: syntax error"
