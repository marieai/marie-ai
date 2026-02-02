"""Tests for marie_wasm.types module."""

import pytest
from marie_wasm.types import (
    BUILTIN_PERMISSIONS,
    COMPILER_CONFIGS,
    CompilerConfig,
    DataItem,
    ExecutionContext,
    ExecutionResult,
    Language,
    Permissions,
)


class TestLanguage:
    """Tests for Language enum."""

    def test_language_values(self):
        """Test language enum values."""
        assert Language.RUST.value == "rust"
        assert Language.PYTHON.value == "python"
        assert Language.JAVASCRIPT.value == "js"

    def test_from_string_valid(self):
        """Test parsing valid language strings."""
        assert Language.from_string("rust") == Language.RUST
        assert Language.from_string("python") == Language.PYTHON
        assert Language.from_string("js") == Language.JAVASCRIPT
        assert Language.from_string("javascript") == Language.JAVASCRIPT
        assert Language.from_string("RUST") == Language.RUST
        assert Language.from_string("  Python  ") == Language.PYTHON

    def test_from_string_aliases(self):
        """Test language aliases."""
        assert Language.from_string("py") == Language.PYTHON
        assert Language.from_string("rs") == Language.RUST
        assert Language.from_string("ts") == Language.JAVASCRIPT
        assert Language.from_string("typescript") == Language.JAVASCRIPT

    def test_from_string_invalid(self):
        """Test parsing invalid language strings."""
        with pytest.raises(ValueError):
            Language.from_string("invalid")
        with pytest.raises(ValueError):
            Language.from_string("")


class TestPermissions:
    """Tests for Permissions dataclass."""

    def test_default_permissions(self):
        """Test default permission values."""
        perms = Permissions()
        assert perms.allow_http is False
        assert perms.allow_secrets is False
        assert perms.allow_kv is False
        assert perms.max_memory_mb == 64
        assert perms.max_fuel == 1_000_000_000
        assert perms.timeout_ms == 30_000

    def test_is_host_allowed_disabled(self):
        """Test host check when HTTP is disabled."""
        perms = Permissions(allow_http=False)
        assert perms.is_host_allowed("example.com") is False

    def test_is_host_allowed_empty_list(self):
        """Test host check with empty allowed list (all allowed)."""
        perms = Permissions(allow_http=True, http_allowed_hosts=[])
        assert perms.is_host_allowed("example.com") is True
        assert perms.is_host_allowed("any.domain.com") is True

    def test_is_host_allowed_specific_hosts(self):
        """Test host check with specific allowed hosts."""
        perms = Permissions(
            allow_http=True,
            http_allowed_hosts=["api.example.com", "github.com"],
        )
        assert perms.is_host_allowed("api.example.com") is True
        assert perms.is_host_allowed("github.com") is True
        assert perms.is_host_allowed("sub.api.example.com") is True  # Subdomain
        assert perms.is_host_allowed("evil.com") is False

    def test_is_secret_allowed_disabled(self):
        """Test secret check when secrets are disabled."""
        perms = Permissions(allow_secrets=False)
        assert perms.is_secret_allowed("API_KEY") is False

    def test_is_secret_allowed_empty_list(self):
        """Test secret check with empty allowed list (all allowed)."""
        perms = Permissions(allow_secrets=True, secret_allowed_names=[])
        assert perms.is_secret_allowed("API_KEY") is True
        assert perms.is_secret_allowed("any_secret") is True

    def test_is_secret_allowed_specific_names(self):
        """Test secret check with specific allowed names."""
        perms = Permissions(
            allow_secrets=True,
            secret_allowed_names=["API_KEY", "DATABASE_URL"],
        )
        assert perms.is_secret_allowed("API_KEY") is True
        assert perms.is_secret_allowed("DATABASE_URL") is True
        assert perms.is_secret_allowed("OTHER_SECRET") is False


class TestExecutionContext:
    """Tests for ExecutionContext dataclass."""

    def test_creation(self):
        """Test creating execution context."""
        ctx = ExecutionContext(
            workflow_id="wf-123",
            execution_id="exec-456",
            node_id="node-789",
            run_index=1,
        )
        assert ctx.workflow_id == "wf-123"
        assert ctx.execution_id == "exec-456"
        assert ctx.node_id == "node-789"
        assert ctx.run_index == 1

    def test_to_dict(self):
        """Test converting context to dict."""
        ctx = ExecutionContext(
            workflow_id="wf-123",
            execution_id="exec-456",
            node_id="node-789",
            run_index=2,
        )
        d = ctx.to_dict()
        assert d["workflow-id"] == "wf-123"
        assert d["execution-id"] == "exec-456"
        assert d["node-id"] == "node-789"
        assert d["run-index"] == 2


class TestDataItem:
    """Tests for DataItem dataclass."""

    def test_json_only(self):
        """Test data item with JSON only."""
        item = DataItem(json_data='{"key": "value"}')
        assert item.json_data == '{"key": "value"}'
        assert item.binary is None

    def test_with_binary(self):
        """Test data item with binary data."""
        item = DataItem(json_data='{}', binary=b"hello")
        assert item.binary == b"hello"

    def test_to_dict(self):
        """Test converting to dict."""
        item = DataItem(json_data='{"test": 1}', binary=b"data")
        d = item.to_dict()
        assert d["json"] == '{"test": 1}'
        assert d["binary"] == b"data"

    def test_from_dict(self):
        """Test creating from dict."""
        d = {"json": '{"test": 1}', "binary": b"data"}
        item = DataItem.from_dict(d)
        assert item.json_data == '{"test": 1}'
        assert item.binary == b"data"


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_ok(self):
        """Test creating success result."""
        items = [DataItem(json_data='{}')]
        result = ExecutionResult.ok(items, fuel_consumed=100)
        assert result.success is True
        assert len(result.data) == 1
        assert result.fuel_consumed == 100

    def test_err(self):
        """Test creating error result."""
        result = ExecutionResult.err("Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert len(result.data) == 0

    def test_to_dict_success(self):
        """Test converting success result to dict."""
        items = [DataItem(json_data='{"test": 1}')]
        result = ExecutionResult.ok(items)
        d = result.to_dict()
        assert d["success"] is True
        assert "data" in d
        assert len(d["data"]) == 1

    def test_to_dict_error(self):
        """Test converting error result to dict."""
        result = ExecutionResult.err("Error message")
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "Error message"


class TestCompilerConfig:
    """Tests for CompilerConfig dataclass."""

    def test_compiler_configs_exist(self):
        """Test that configs exist for all languages."""
        assert Language.RUST in COMPILER_CONFIGS
        assert Language.PYTHON in COMPILER_CONFIGS
        assert Language.JAVASCRIPT in COMPILER_CONFIGS

    def test_rust_config(self):
        """Test Rust compiler config."""
        config = COMPILER_CONFIGS[Language.RUST]
        assert config.source_filename == "lib.rs"
        assert "Cargo.toml" in config.additional_files
        assert config.image_suffix == "rust"

    def test_python_config(self):
        """Test Python compiler config."""
        config = COMPILER_CONFIGS[Language.PYTHON]
        assert config.source_filename == "main.py"
        assert config.image_suffix == "python"

    def test_js_config(self):
        """Test JavaScript compiler config."""
        config = COMPILER_CONFIGS[Language.JAVASCRIPT]
        assert config.source_filename == "main.js"
        assert "package.json" in config.additional_files
        assert config.image_suffix == "js"


class TestBuiltinPermissions:
    """Tests for built-in node permissions."""

    def test_http_request_permissions(self):
        """Test HTTP request node permissions."""
        perms = BUILTIN_PERMISSIONS.get("http-request")
        assert perms is not None
        assert perms.allow_http is True
        assert perms.allow_secrets is True

    def test_slack_permissions(self):
        """Test Slack node permissions."""
        perms = BUILTIN_PERMISSIONS.get("slack")
        assert perms is not None
        assert perms.allow_http is True
        assert "slack.com" in perms.http_allowed_hosts
        assert "slack_oauth_token" in perms.secret_allowed_names
