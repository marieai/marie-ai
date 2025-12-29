"""
LLM Tracking Configuration - Environment-based settings.

Configuration follows marie-ai patterns using pydantic-settings.
All settings can be overridden via environment variables with
MARIE_LLM_TRACKING_ prefix.
"""

import os
from enum import Enum
from typing import Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class ExporterType(str, Enum):
    """Available exporter types for LLM tracking events."""

    CONSOLE = "console"
    RABBITMQ = "rabbitmq"


class LLMTrackingSettings(BaseSettings):
    """
    Configuration for LLM tracking module.

    All settings can be overridden via environment variables:
    - MARIE_LLM_TRACKING_ENABLED
    - MARIE_LLM_TRACKING_EXPORTER
    - MARIE_LLM_TRACKING_PROJECT_ID
    - etc.
    """

    # Feature toggle
    ENABLED: bool = Field(
        default=False,
        description="Enable/disable LLM tracking globally",
    )

    # Exporter configuration
    EXPORTER: ExporterType = Field(
        default=ExporterType.CONSOLE,
        description="Exporter type: 'console' for dev, 'rabbitmq' for prod",
    )

    # Project configuration
    PROJECT_ID: str = Field(
        default="default",
        description="Default project ID for traces",
    )

    # Batching configuration
    BATCH_SIZE: int = Field(
        default=100,
        description="Number of events to batch before flushing",
    )
    FLUSH_INTERVAL_SECONDS: float = Field(
        default=5.0,
        description="Maximum time between flushes in seconds",
    )

    # Payload storage thresholds
    PAYLOAD_SIZE_THRESHOLD_BYTES: int = Field(
        default=100_000,  # 100KB
        description="Payloads larger than this go to S3, smaller stay in Postgres",
    )

    # PostgreSQL configuration
    POSTGRES_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection URL for raw event storage",
    )

    # S3 configuration
    S3_BUCKET: Optional[str] = Field(
        default=None,
        description="S3 bucket for large payload storage",
    )
    S3_ENDPOINT: Optional[str] = Field(
        default=None,
        description="S3 endpoint URL (for MinIO compatibility)",
    )
    S3_REGION: str = Field(
        default="us-east-1",
        description="S3 region",
    )
    S3_ACCESS_KEY: Optional[str] = Field(
        default=None,
        description="S3 access key (if not using IAM)",
    )
    S3_SECRET_KEY: Optional[str] = Field(
        default=None,
        description="S3 secret key (if not using IAM)",
    )

    # RabbitMQ configuration
    RABBITMQ_URL: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        description="RabbitMQ connection URL",
    )
    RABBITMQ_EXCHANGE: str = Field(
        default="llm-events",
        description="RabbitMQ exchange name for LLM events",
    )
    RABBITMQ_QUEUE: str = Field(
        default="llm-ingestion",
        description="RabbitMQ queue name for LLM event processing",
    )
    RABBITMQ_ROUTING_KEY: str = Field(
        default="llm.event",
        description="RabbitMQ routing key for events",
    )

    # ClickHouse configuration (for worker)
    CLICKHOUSE_HOST: str = Field(
        default="localhost",
        description="ClickHouse server host",
    )
    CLICKHOUSE_PORT: int = Field(
        default=8123,
        description="ClickHouse HTTP port",
    )
    CLICKHOUSE_NATIVE_PORT: int = Field(
        default=9000,
        description="ClickHouse native protocol port",
    )
    CLICKHOUSE_DATABASE: str = Field(
        default="marie_llm",
        description="ClickHouse database name",
    )
    CLICKHOUSE_USER: str = Field(
        default="default",
        description="ClickHouse username",
    )
    CLICKHOUSE_PASSWORD: str = Field(
        default="",
        description="ClickHouse password",
    )

    # ClickHouse writer configuration
    CLICKHOUSE_BATCH_SIZE: int = Field(
        default=1000,
        description="Number of records to batch before writing to ClickHouse",
    )
    CLICKHOUSE_FLUSH_INTERVAL_S: float = Field(
        default=5.0,
        description="Maximum time between ClickHouse flushes in seconds",
    )
    CLICKHOUSE_MAX_ATTEMPTS: int = Field(
        default=3,
        description="Maximum retry attempts for ClickHouse writes",
    )

    # Token counting configuration
    TOKEN_COUNTING_ENABLED: bool = Field(
        default=True,
        description="Enable tiktoken-based token counting fallback",
    )
    DEFAULT_TOKENIZER_MODEL: str = Field(
        default="gpt-4",
        description="Default model for tiktoken encoding when model unknown",
    )

    # Sampling configuration (for high-volume scenarios)
    SAMPLING_RATE: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sampling rate for traces (1.0 = 100%, 0.1 = 10%)",
    )

    # Debug configuration
    DEBUG: bool = Field(
        default=False,
        description="Enable debug logging for LLM tracking",
    )
    SAVE_DEBUG_PAYLOADS: bool = Field(
        default=False,
        description="Save raw payloads to /tmp for debugging",
    )

    @computed_field
    @property
    def clickhouse_url(self) -> str:
        """Construct ClickHouse connection URL."""
        password_part = (
            f":{self.CLICKHOUSE_PASSWORD}@" if self.CLICKHOUSE_PASSWORD else "@"
        )
        return f"clickhouse://{self.CLICKHOUSE_USER}{password_part}{self.CLICKHOUSE_HOST}:{self.CLICKHOUSE_NATIVE_PORT}/{self.CLICKHOUSE_DATABASE}"

    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production mode (RabbitMQ exporter)."""
        return self.EXPORTER == ExporterType.RABBITMQ

    @computed_field
    @property
    def storage_enabled(self) -> bool:
        """Check if durable storage (Postgres + S3) is configured."""
        return bool(self.POSTGRES_URL)

    class Config:
        env_prefix = "MARIE_LLM_TRACKING_"
        env_file = ".env"
        extra = "ignore"


# Singleton instance
_settings: Optional[LLMTrackingSettings] = None


def get_settings() -> LLMTrackingSettings:
    """Get or create the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = LLMTrackingSettings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
