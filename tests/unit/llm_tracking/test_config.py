"""
Tests for LLM tracking configuration module.

Tests configuration loading, validation, and URL construction.
"""

import pytest

from marie.llm_tracking.config import (
    ExporterType,
    LLMTrackingSettings,
    configure_from_yaml,
    get_settings,
    reset_settings,
)


@pytest.fixture(autouse=True)
def reset_settings_between_tests():
    """Reset settings before and after each test."""
    reset_settings()
    yield
    reset_settings()


class TestConfigureFromYamlMinimal:
    """Test config with minimal required settings."""

    def test_configure_from_yaml_minimal(self):
        """Test config with minimal required settings returns defaults."""
        config = {"enabled": True}
        settings = configure_from_yaml(config)

        assert settings.ENABLED is True
        assert settings.EXPORTER == ExporterType.CONSOLE
        assert settings.PROJECT_ID == "default"
        assert settings.BATCH_SIZE == 100
        assert settings.FLUSH_INTERVAL_SECONDS == 5.0

    def test_configure_from_yaml_empty_config(self):
        """Test config with empty dict uses all defaults."""
        config = {}
        settings = configure_from_yaml(config)

        assert settings.ENABLED is False
        assert settings.EXPORTER == ExporterType.CONSOLE


class TestConfigureFromYamlFull:
    """Test config with all settings specified."""

    def test_configure_from_yaml_full(self):
        """Test config with all settings specified."""
        config = {
            "enabled": True,
            "exporter": "rabbitmq",
            "project_id": "test-project",
            "debug": True,
            "sampling_rate": 0.5,
            "batch_size": 200,
            "flush_interval_seconds": 10.0,
            "rabbitmq": {
                "url": "amqp://user:pass@host:5672/myvhost",
                "exchange": "test-exchange",
                "queue": "test-queue",
                "routing_key": "test.key",
            },
            "postgres": {
                "url": "postgresql://user:pass@host:5432/testdb",
            },
            "s3": {
                "bucket": "test-bucket",
            },
            "clickhouse": {
                "host": "ch-host",
                "port": 8124,
                "native_port": 9001,
                "database": "test_db",
                "user": "ch_user",
                "password": "ch_pass",
                "batch_size": 500,
                "flush_interval_s": 2.0,
                "max_attempts": 5,
            },
        }
        settings = configure_from_yaml(config)

        assert settings.ENABLED is True
        assert settings.EXPORTER == ExporterType.RABBITMQ
        assert settings.PROJECT_ID == "test-project"
        assert settings.DEBUG is True
        assert settings.SAMPLING_RATE == 0.5
        assert settings.BATCH_SIZE == 200
        assert settings.FLUSH_INTERVAL_SECONDS == 10.0
        assert settings.RABBITMQ_URL == "amqp://user:pass@host:5672/myvhost"
        assert settings.RABBITMQ_EXCHANGE == "test-exchange"
        assert settings.RABBITMQ_QUEUE == "test-queue"
        assert settings.RABBITMQ_ROUTING_KEY == "test.key"
        assert settings.POSTGRES_URL == "postgresql://user:pass@host:5432/testdb"
        assert settings.S3_BUCKET == "test-bucket"
        assert settings.CLICKHOUSE_HOST == "ch-host"
        assert settings.CLICKHOUSE_PORT == 8124
        assert settings.CLICKHOUSE_NATIVE_PORT == 9001
        assert settings.CLICKHOUSE_DATABASE == "test_db"
        assert settings.CLICKHOUSE_USER == "ch_user"
        assert settings.CLICKHOUSE_PASSWORD == "ch_pass"
        assert settings.CLICKHOUSE_BATCH_SIZE == 500
        assert settings.CLICKHOUSE_FLUSH_INTERVAL_S == 2.0
        assert settings.CLICKHOUSE_MAX_ATTEMPTS == 5


class TestVhostKeySupport:
    """Test both 'vhost' and 'virtualhost' keys work."""

    def test_vhost_key_works(self):
        """Test 'vhost' key is properly recognized."""
        config = {
            "rabbitmq": {
                "hostname": "localhost",
                "username": "user",
                "password": "pass",
                "vhost": "myvhost",
            }
        }
        settings = configure_from_yaml(config)

        assert "myvhost" in settings.RABBITMQ_URL
        assert settings.RABBITMQ_URL == "amqp://user:pass@localhost:5672/myvhost"

    def test_virtualhost_key_works(self):
        """Test 'virtualhost' key is properly recognized (legacy)."""
        config = {
            "rabbitmq": {
                "hostname": "localhost",
                "username": "user",
                "password": "pass",
                "virtualhost": "legacyvhost",
            }
        }
        settings = configure_from_yaml(config)

        assert "legacyvhost" in settings.RABBITMQ_URL
        assert settings.RABBITMQ_URL == "amqp://user:pass@localhost:5672/legacyvhost"

    def test_vhost_takes_precedence_over_virtualhost(self):
        """Test 'vhost' takes precedence when both are specified."""
        config = {
            "rabbitmq": {
                "hostname": "localhost",
                "username": "user",
                "password": "pass",
                "vhost": "preferred",
                "virtualhost": "ignored",
            }
        }
        settings = configure_from_yaml(config)

        assert "preferred" in settings.RABBITMQ_URL
        assert "ignored" not in settings.RABBITMQ_URL

    def test_default_vhost_is_slash(self):
        """Test default vhost is '/' when neither key specified."""
        config = {
            "rabbitmq": {
                "hostname": "localhost",
                "username": "user",
                "password": "pass",
            }
        }
        settings = configure_from_yaml(config)

        assert settings.RABBITMQ_URL == "amqp://user:pass@localhost:5672/"


class TestRabbitMQUrlConstruction:
    """Test URL is built correctly from individual fields."""

    def test_url_from_individual_fields(self):
        """Test RabbitMQ URL construction from individual fields."""
        config = {
            "rabbitmq": {
                "hostname": "rabbit.example.com",
                "port": 5673,
                "username": "myuser",
                "password": "mypass",
                "vhost": "production",
            }
        }
        settings = configure_from_yaml(config)

        assert settings.RABBITMQ_URL == "amqp://myuser:mypass@rabbit.example.com:5673/production"

    def test_url_with_tls(self):
        """Test RabbitMQ URL with TLS enabled."""
        config = {
            "rabbitmq": {
                "hostname": "rabbit.example.com",
                "tls": True,
                "username": "user",
                "password": "pass",
            }
        }
        settings = configure_from_yaml(config)

        assert settings.RABBITMQ_URL.startswith("amqps://")

    def test_url_with_special_chars_in_vhost(self):
        """Test vhost with special characters is URL-encoded."""
        config = {
            "rabbitmq": {
                "hostname": "localhost",
                "username": "user",
                "password": "pass",
                "vhost": "my/vhost",
            }
        }
        settings = configure_from_yaml(config)

        # The vhost should be URL-encoded
        assert "my%2Fvhost" in settings.RABBITMQ_URL

    def test_direct_url_takes_precedence(self):
        """Test direct URL takes precedence over individual fields."""
        config = {
            "rabbitmq": {
                "url": "amqp://direct:url@host:1234/vhost",
                "hostname": "ignored",
                "username": "ignored",
            }
        }
        settings = configure_from_yaml(config)

        assert settings.RABBITMQ_URL == "amqp://direct:url@host:1234/vhost"


class TestPostgresUrlConstruction:
    """Test PostgreSQL URL is built correctly."""

    def test_url_from_individual_fields(self):
        """Test Postgres URL construction from individual fields."""
        config = {
            "postgres": {
                "hostname": "pg.example.com",
                "port": 5433,
                "database": "mydb",
                "username": "pguser",
                "password": "pgpass",
            }
        }
        settings = configure_from_yaml(config)

        assert settings.POSTGRES_URL == "postgresql://pguser:pgpass@pg.example.com:5433/mydb"

    def test_url_with_defaults(self):
        """Test Postgres URL with default values."""
        config = {
            "postgres": {
                "hostname": "localhost",
            }
        }
        settings = configure_from_yaml(config)

        assert settings.POSTGRES_URL == "postgresql://postgres:@localhost:5432/marie"

    def test_direct_url_takes_precedence(self):
        """Test direct URL takes precedence over individual fields."""
        config = {
            "postgres": {
                "url": "postgresql://direct:url@host:1234/db",
                "hostname": "ignored",
            }
        }
        settings = configure_from_yaml(config)

        assert settings.POSTGRES_URL == "postgresql://direct:url@host:1234/db"


class TestExporterTypeValidation:
    """Test invalid exporter type raises error."""

    def test_invalid_exporter_type_raises_error(self):
        """Test invalid exporter type raises ValueError."""
        config = {"exporter": "invalid_exporter"}

        with pytest.raises(ValueError):
            configure_from_yaml(config)

    def test_valid_console_exporter(self):
        """Test console exporter is valid."""
        config = {"exporter": "console"}
        settings = configure_from_yaml(config)
        assert settings.EXPORTER == ExporterType.CONSOLE

    def test_valid_rabbitmq_exporter(self):
        """Test rabbitmq exporter is valid."""
        config = {"exporter": "rabbitmq"}
        settings = configure_from_yaml(config)
        assert settings.EXPORTER == ExporterType.RABBITMQ


class TestGetSettingsRaises:
    """Test get_settings() raises if configure_from_yaml not called."""

    def test_get_settings_raises_without_configure(self):
        """Test get_settings() raises RuntimeError if not configured."""
        # Settings already reset by fixture
        with pytest.raises(RuntimeError) as exc_info:
            get_settings()

        assert "not configured" in str(exc_info.value)
        assert "configure_from_yaml" in str(exc_info.value)

    def test_get_settings_returns_after_configure(self):
        """Test get_settings() returns settings after configure."""
        configure_from_yaml({"enabled": True})
        settings = get_settings()

        assert settings is not None
        assert settings.ENABLED is True


class TestS3BucketFromStorageConfig:
    """Test S3 bucket can be loaded from shared storage config."""

    def test_s3_bucket_from_llm_tracking_config(self):
        """Test S3 bucket from llm_tracking.s3 section."""
        config = {"s3": {"bucket": "tracking-bucket"}}
        settings = configure_from_yaml(config)

        assert settings.S3_BUCKET == "tracking-bucket"

    def test_s3_bucket_from_storage_config(self):
        """Test S3 bucket falls back to storage section."""
        config = {}
        storage_config = {"s3": {"bucket_name": "shared-bucket"}}
        settings = configure_from_yaml(config, storage_config)

        assert settings.S3_BUCKET == "shared-bucket"

    def test_llm_tracking_s3_takes_precedence(self):
        """Test llm_tracking.s3 takes precedence over storage.s3."""
        config = {"s3": {"bucket": "tracking-bucket"}}
        storage_config = {"s3": {"bucket_name": "shared-bucket"}}
        settings = configure_from_yaml(config, storage_config)

        assert settings.S3_BUCKET == "tracking-bucket"


class TestSettingsProperties:
    """Test computed properties on LLMTrackingSettings."""

    def test_is_production_true_for_rabbitmq(self):
        """Test is_production returns True for RabbitMQ exporter."""
        config = {"exporter": "rabbitmq"}
        settings = configure_from_yaml(config)

        assert settings.is_production is True

    def test_is_production_false_for_console(self):
        """Test is_production returns False for console exporter."""
        config = {"exporter": "console"}
        settings = configure_from_yaml(config)

        assert settings.is_production is False

    def test_storage_enabled_with_s3(self):
        """Test storage_enabled returns True when S3 configured."""
        config = {"s3": {"bucket": "test-bucket"}}
        settings = configure_from_yaml(config)

        assert settings.storage_enabled is True

    def test_storage_disabled_without_s3(self):
        """Test storage_enabled returns False when S3 not configured."""
        config = {}
        settings = configure_from_yaml(config)

        assert settings.storage_enabled is False

    def test_clickhouse_url_construction(self):
        """Test ClickHouse URL is constructed correctly."""
        config = {
            "clickhouse": {
                "host": "ch-host",
                "native_port": 9001,
                "database": "test_db",
                "user": "ch_user",
                "password": "ch_pass",
            }
        }
        settings = configure_from_yaml(config)

        assert settings.clickhouse_url == "clickhouse://ch_user:ch_pass@ch-host:9001/test_db"

    def test_clickhouse_url_without_password(self):
        """Test ClickHouse URL without password."""
        config = {
            "clickhouse": {
                "host": "ch-host",
                "native_port": 9001,
                "database": "test_db",
                "user": "ch_user",
            }
        }
        settings = configure_from_yaml(config)

        assert settings.clickhouse_url == "clickhouse://ch_user@ch-host:9001/test_db"
