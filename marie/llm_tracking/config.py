"""
LLM Tracking Configuration - Unified config from YAML.

Configuration is ONLY loaded from YAML config via configure_from_yaml().
There is no fallback to environment variables - YAML is the single source of truth.
"""

import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExporterType(str, Enum):
    """Available exporter types for LLM tracking events."""

    CONSOLE = "console"
    RABBITMQ = "rabbitmq"


@dataclass
class LLMTrackingSettings:
    """
    Configuration for LLM tracking module.

    Must be initialized from YAML config via configure_from_yaml().
    There is no fallback - config must be explicitly provided.
    """

    # Feature toggle
    ENABLED: bool = False

    # Exporter configuration
    EXPORTER: ExporterType = ExporterType.CONSOLE

    # Project configuration
    PROJECT_ID: str = "default"

    # Batching configuration
    BATCH_SIZE: int = 100
    FLUSH_INTERVAL_SECONDS: float = 5.0

    # PostgreSQL configuration (metadata only)
    POSTGRES_URL: Optional[str] = None

    # S3 configuration (all payloads stored here)
    # Note: S3 credentials are managed by StorageManager's S3StorageHandler
    S3_BUCKET: Optional[str] = None

    # RabbitMQ configuration
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
    RABBITMQ_EXCHANGE: str = "llm-events"
    RABBITMQ_QUEUE: str = "llm-ingestion"
    RABBITMQ_ROUTING_KEY: str = "llm.event"

    # ClickHouse configuration (for worker)
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 8123
    CLICKHOUSE_NATIVE_PORT: int = 9000
    CLICKHOUSE_DATABASE: str = "marie"
    CLICKHOUSE_USER: str = "default"
    CLICKHOUSE_PASSWORD: str = ""

    # ClickHouse writer configuration
    CLICKHOUSE_BATCH_SIZE: int = 1000
    CLICKHOUSE_FLUSH_INTERVAL_S: float = 5.0
    CLICKHOUSE_MAX_ATTEMPTS: int = 3

    # Token counting configuration
    TOKEN_COUNTING_ENABLED: bool = True
    DEFAULT_TOKENIZER_MODEL: str = "gpt-4"

    # Sampling configuration (for high-volume scenarios)
    SAMPLING_RATE: float = 1.0

    # Debug configuration
    DEBUG: bool = False
    SAVE_DEBUG_PAYLOADS: bool = False

    @property
    def clickhouse_url(self) -> str:
        """Construct ClickHouse connection URL."""
        password_part = (
            f":{self.CLICKHOUSE_PASSWORD}@" if self.CLICKHOUSE_PASSWORD else "@"
        )
        return f"clickhouse://{self.CLICKHOUSE_USER}{password_part}{self.CLICKHOUSE_HOST}:{self.CLICKHOUSE_NATIVE_PORT}/{self.CLICKHOUSE_DATABASE}"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode (RabbitMQ exporter)."""
        return self.EXPORTER == ExporterType.RABBITMQ

    @property
    def storage_enabled(self) -> bool:
        """Check if durable storage is configured (S3 required, Postgres optional)."""
        return bool(self.S3_BUCKET)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        storage_config: Optional[Dict[str, Any]] = None,
    ) -> "LLMTrackingSettings":
        """
        Create settings from YAML config dict.

        Args:
            config: The llm_tracking section from YAML config
            storage_config: Optional storage section to get S3 bucket from shared config

        Returns:
            LLMTrackingSettings instance
        """
        kwargs = {}

        # Map simple fields
        if "enabled" in config:
            kwargs["ENABLED"] = config["enabled"]
        if "exporter" in config:
            kwargs["EXPORTER"] = ExporterType(config["exporter"])
        if "project_id" in config:
            kwargs["PROJECT_ID"] = config["project_id"]
        if "debug" in config:
            kwargs["DEBUG"] = config["debug"]
        if "sampling_rate" in config:
            kwargs["SAMPLING_RATE"] = config["sampling_rate"]
        if "batch_size" in config:
            kwargs["BATCH_SIZE"] = config["batch_size"]
        if "flush_interval_seconds" in config:
            kwargs["FLUSH_INTERVAL_SECONDS"] = config["flush_interval_seconds"]

        # RabbitMQ config
        if "rabbitmq" in config:
            rabbitmq = config["rabbitmq"]
            if "url" in rabbitmq:
                kwargs["RABBITMQ_URL"] = rabbitmq["url"]
            elif "hostname" in rabbitmq:
                # Construct URL from individual fields (matches marie-ai shared config)
                scheme = "amqps" if rabbitmq.get("tls", False) else "amqp"
                username = rabbitmq.get("username", "guest")
                password = rabbitmq.get("password", "guest")
                hostname = rabbitmq["hostname"]
                port = rabbitmq.get("port", 5672)
                # Support both 'vhost' (YAML spec) and 'virtualhost' (legacy)
                vhost = rabbitmq.get("vhost", rabbitmq.get("virtualhost", "/"))
                # URL-encode vhost for non-default vhosts
                # Default "/" becomes trailing "/" in URL
                # Non-default "marie" becomes "/marie"
                # Vhosts with special chars get URL-encoded
                if vhost == "/":
                    vhost_part = "/"
                else:
                    # URL-encode the vhost but keep it after the port
                    vhost_part = "/" + urllib.parse.quote(vhost, safe="")
                kwargs["RABBITMQ_URL"] = (
                    f"{scheme}://{username}:{password}@{hostname}:{port}{vhost_part}"
                )
            if "exchange" in rabbitmq:
                kwargs["RABBITMQ_EXCHANGE"] = rabbitmq["exchange"]
            if "queue" in rabbitmq:
                kwargs["RABBITMQ_QUEUE"] = rabbitmq["queue"]
            if "routing_key" in rabbitmq:
                kwargs["RABBITMQ_ROUTING_KEY"] = rabbitmq["routing_key"]

        # ClickHouse config
        if "clickhouse" in config:
            ch = config["clickhouse"]
            if "host" in ch:
                kwargs["CLICKHOUSE_HOST"] = ch["host"]
            if "port" in ch:
                kwargs["CLICKHOUSE_PORT"] = ch["port"]
            if "native_port" in ch:
                kwargs["CLICKHOUSE_NATIVE_PORT"] = ch["native_port"]
            if "database" in ch:
                kwargs["CLICKHOUSE_DATABASE"] = ch["database"]
            if "user" in ch:
                kwargs["CLICKHOUSE_USER"] = ch["user"]
            if "password" in ch:
                kwargs["CLICKHOUSE_PASSWORD"] = ch["password"]
            if "batch_size" in ch:
                kwargs["CLICKHOUSE_BATCH_SIZE"] = ch["batch_size"]
            if "flush_interval_s" in ch:
                kwargs["CLICKHOUSE_FLUSH_INTERVAL_S"] = ch["flush_interval_s"]
            if "max_attempts" in ch:
                kwargs["CLICKHOUSE_MAX_ATTEMPTS"] = ch["max_attempts"]

        # PostgreSQL config
        if "postgres" in config:
            pg = config["postgres"]
            if "url" in pg:
                kwargs["POSTGRES_URL"] = pg["url"]
            elif "hostname" in pg:
                # Construct URL from individual fields
                username = pg.get("username", "postgres")
                password = pg.get("password", "")
                hostname = pg["hostname"]
                port = pg.get("port", 5432)
                database = pg.get("database", "marie")
                kwargs["POSTGRES_URL"] = (
                    f"postgresql://{username}:{password}@{hostname}:{port}/{database}"
                )

        # S3 config - prefer from llm_tracking.s3, fall back to shared storage.s3
        if "s3" in config:
            s3 = config["s3"]
            if "bucket" in s3:
                kwargs["S3_BUCKET"] = s3["bucket"]
        elif storage_config and "s3" in storage_config:
            # Use shared storage config
            s3 = storage_config["s3"]
            if "bucket_name" in s3:
                kwargs["S3_BUCKET"] = s3["bucket_name"]
            elif "bucket" in s3:
                kwargs["S3_BUCKET"] = s3["bucket"]

        return cls(**kwargs)


# Singleton instance - must be configured via configure_from_yaml()
_settings: Optional[LLMTrackingSettings] = None


def get_settings() -> LLMTrackingSettings:
    """
    Get the configured settings instance.

    Raises:
        RuntimeError: If settings have not been configured via configure_from_yaml()
    """
    if _settings is None:
        raise RuntimeError(
            "LLM tracking settings not configured. "
            "Call configure_from_yaml() first or ensure llm_tracking section "
            "is present in the YAML config."
        )
    return _settings


def configure_from_yaml(
    config: Dict[str, Any],
    storage_config: Optional[Dict[str, Any]] = None,
) -> LLMTrackingSettings:
    """
    Configure settings from YAML config (required).

    This is the ONLY way to initialize settings - there is no fallback.

    Args:
        config: The llm_tracking section from YAML config
        storage_config: Optional storage section for shared S3 config

    Returns:
        The configured LLMTrackingSettings instance
    """
    global _settings
    _settings = LLMTrackingSettings.from_config(config, storage_config)
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
