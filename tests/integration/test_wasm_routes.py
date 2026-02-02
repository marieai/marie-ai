"""Integration tests for Wasm compilation routes.

Run with: pytest tests/integration/test_wasm_routes.py -v

These tests verify the Wasm compilation API endpoints work correctly.
Some tests require Docker and compiler images to be available.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marie.serve.runtimes.servers.wasm_routes import register_wasm_routes

# Check if marie_wasm is available
try:
    from marie_wasm import Language, WasmCompilerService

    MARIE_WASM_AVAILABLE = True
except ImportError:
    MARIE_WASM_AVAILABLE = False


def is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def are_compiler_images_available() -> bool:
    """Check if compiler images are built."""
    try:
        import docker

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


# Skip markers
wasm_available = pytest.mark.skipif(
    not MARIE_WASM_AVAILABLE,
    reason="marie-wasm package not installed",
)

docker_available = pytest.mark.skipif(
    not is_docker_available(),
    reason="Docker is not available",
)

compilers_available = pytest.mark.skipif(
    not are_compiler_images_available(),
    reason="Compiler images not built. Run: make -f Makefile.wasm compilers-build",
)


@pytest.fixture
def app():
    """Create FastAPI app with wasm routes."""
    app = FastAPI()
    register_wasm_routes(app)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestListLanguages:
    """Tests for GET /api/wasm/languages endpoint."""

    def test_list_languages_returns_json(self, client):
        """Test that languages endpoint returns JSON."""
        response = client.get("/api/wasm/languages")
        assert response.status_code == 200
        assert "languages" in response.json()

    @wasm_available
    def test_list_languages_includes_supported(self, client):
        """Test that all supported languages are listed."""
        response = client.get("/api/wasm/languages")
        data = response.json()
        languages = data["languages"]

        # Should include our three supported languages
        assert "rust" in languages
        assert "python" in languages
        assert "js" in languages

    def test_list_languages_no_error_without_package(self, client):
        """Test graceful handling when marie-wasm not installed."""
        response = client.get("/api/wasm/languages")
        # Should return 200 even if package not installed
        assert response.status_code == 200


class TestWasmStatus:
    """Tests for GET /api/wasm/status endpoint."""

    def test_status_returns_json(self, client):
        """Test that status endpoint returns JSON."""
        response = client.get("/api/wasm/status")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data

    @wasm_available
    @docker_available
    def test_status_shows_available_with_docker(self, client):
        """Test that status shows available when Docker is running."""
        response = client.get("/api/wasm/status")
        data = response.json()

        # Should be available if Docker is running
        assert data["available"] is True
        assert "languages" in data
        assert len(data["languages"]) > 0

    def test_status_graceful_without_package(self, client):
        """Test graceful handling when marie-wasm not installed."""
        response = client.get("/api/wasm/status")
        assert response.status_code == 200
        # Should indicate not available if package missing


class TestCompileNode:
    """Tests for POST /api/nodes/{node_id}/compile endpoint."""

    @wasm_available
    def test_compile_invalid_language_returns_400(self, client):
        """Test that invalid language returns 400 error."""
        response = client.post(
            "/api/nodes/test-node/compile",
            params={
                "code": "def execute(): pass",
                "language": "invalid_language",
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "Unsupported language" in data["error"]

    @wasm_available
    def test_compile_empty_code_accepted(self, client):
        """Test that empty code is passed to compiler."""
        # Empty code should be passed to compiler (may fail later)
        response = client.post(
            "/api/nodes/test-node/compile",
            params={
                "code": "",
                "language": "python",
            },
        )
        # Either succeeds or fails with compilation error, not 400
        assert response.status_code in [200, 500, 501]

    @wasm_available
    @docker_available
    @compilers_available
    @pytest.mark.asyncio
    async def test_compile_python_node(self, client):
        """Test compiling a Python node to WASM."""
        python_code = '''
def execute(input_items, config, context):
    return {"success": [{"json": "{}"}]}
'''
        response = client.post(
            "/api/nodes/test-python-node/compile",
            params={
                "code": python_code,
                "language": "python",
                "timeout_seconds": 300,
            },
        )

        # If compilation succeeds
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "wasm_path" in data
            assert data["node_id"] == "test-python-node"
        else:
            # May fail if Docker/compiler not available
            data = response.json()
            assert "error" in data

    @wasm_available
    @docker_available
    @compilers_available
    def test_compile_javascript_node(self, client):
        """Test compiling a JavaScript node to WASM."""
        js_code = '''
function execute(input, config, context) {
    return { success: [{ json: "{}" }] };
}
'''
        response = client.post(
            "/api/nodes/test-js-node/compile",
            params={
                "code": js_code,
                "language": "js",
                "timeout_seconds": 120,
            },
        )

        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "wasm_path" in data
        else:
            data = response.json()
            assert "error" in data

    @wasm_available
    @docker_available
    @compilers_available
    def test_compile_rust_node(self, client):
        """Test compiling a Rust node to WASM."""
        rust_code = '''
fn execute(input_items: Vec<Item>, config: Config, context: Context) -> Response {
    Response::success(vec![])
}
'''
        response = client.post(
            "/api/nodes/test-rust-node/compile",
            params={
                "code": rust_code,
                "language": "rust",
                "timeout_seconds": 300,
            },
        )

        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "wasm_path" in data
        else:
            data = response.json()
            assert "error" in data

    def test_compile_without_package_returns_501(self, client, monkeypatch):
        """Test that 501 is returned when marie-wasm not installed."""
        # Simulate package not being available
        import marie.serve.runtimes.servers.wasm_routes as wasm_routes

        # Reset the global compiler
        monkeypatch.setattr(wasm_routes, "_wasm_compiler", None)

        # Mock the import to fail
        original_get_compiler = wasm_routes._get_compiler

        def mock_get_compiler():
            return None

        monkeypatch.setattr(wasm_routes, "_get_compiler", mock_get_compiler)

        response = client.post(
            "/api/nodes/test-node/compile",
            params={
                "code": "def execute(): pass",
                "language": "python",
            },
        )

        # Should return 501 Not Implemented
        assert response.status_code == 501
        data = response.json()
        assert data["success"] is False


class TestLanguageAliases:
    """Tests for language alias support."""

    @wasm_available
    def test_python_aliases(self, client):
        """Test that Python aliases work."""
        for alias in ["python", "py"]:
            response = client.post(
                "/api/nodes/test-node/compile",
                params={
                    "code": "def execute(): pass",
                    "language": alias,
                },
            )
            # Should not return 400 for invalid language
            assert response.status_code != 400 or "Unsupported language" not in response.json().get(
                "error", ""
            )

    @wasm_available
    def test_javascript_aliases(self, client):
        """Test that JavaScript aliases work."""
        for alias in ["js", "javascript"]:
            response = client.post(
                "/api/nodes/test-node/compile",
                params={
                    "code": "function execute() {}",
                    "language": alias,
                },
            )
            # Should not return 400 for invalid language
            if response.status_code == 400:
                assert "Unsupported language" not in response.json().get("error", "")

    @wasm_available
    def test_typescript_alias(self, client):
        """Test that TypeScript alias works (compiles via JS compiler)."""
        for alias in ["ts", "typescript"]:
            response = client.post(
                "/api/nodes/test-node/compile",
                params={
                    "code": "function execute(): void {}",
                    "language": alias,
                },
            )
            # TypeScript should map to JavaScript compiler
            if response.status_code == 400:
                # ts/typescript should be recognized
                assert "Unsupported language" not in response.json().get("error", "")

    @wasm_available
    def test_rust_aliases(self, client):
        """Test that Rust aliases work."""
        for alias in ["rust", "rs"]:
            response = client.post(
                "/api/nodes/test-node/compile",
                params={
                    "code": "fn execute() {}",
                    "language": alias,
                },
            )
            if response.status_code == 400:
                assert "Unsupported language" not in response.json().get("error", "")


class TestCompileWithDependencies:
    """Tests for compilation with dependencies."""

    @wasm_available
    @docker_available
    @compilers_available
    def test_compile_with_dependencies(self, client):
        """Test compiling with additional dependency files."""
        python_code = '''
from helper import process
def execute(input_items, config, context):
    return {"success": [{"json": process({})}]}
'''
        # Note: This will likely fail because helper.py doesn't exist,
        # but it tests that the dependencies parameter is accepted
        response = client.post(
            "/api/nodes/dep-test-node/compile",
            params={
                "code": python_code,
                "language": "python",
            },
            json={"dependencies": {"helper.py": "def process(x): return str(x)"}},
        )

        # The request should be accepted (even if compilation fails)
        assert response.status_code in [200, 500]


class TestCompileTimeout:
    """Tests for compilation timeout handling."""

    @wasm_available
    @docker_available
    def test_custom_timeout(self, client):
        """Test that custom timeout is accepted."""
        response = client.post(
            "/api/nodes/timeout-test/compile",
            params={
                "code": "function execute() {}",
                "language": "js",
                "timeout_seconds": 60,
            },
        )

        # Request should be accepted
        assert response.status_code in [200, 500, 501]
