"""Runtime configuration for Marie instrumentation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ExporterType(str, Enum):
    """Available exporter types for instrumentation events."""

    CONSOLE = "console"
    OTEL = "otel"


@dataclass
class InstrumentationSettings:
    """
    Configuration for the instrumentation module.

    Must be initialized explicitly from the host application's configuration.
    """

    # Feature toggle
    ENABLED: bool = False

    # Exporter configuration
    EXPORTER: ExporterType = ExporterType.CONSOLE

    # Project configuration
    PROJECT_ID: str = "default"

    # Token counting configuration
    TOKEN_COUNTING_ENABLED: bool = True
    DEFAULT_TOKENIZER_MODEL: str = "gpt-4"

    # console exporter only; ignored/deprecated for exporter=otel
    SAMPLING_RATE: float = 1.0

    # When True, adds ConsoleSpanExporter so OTel spans print to stdout.
    # Auto-enabled when EXPORTER is CONSOLE.
    CONSOLE_SPANS: bool = False

    # Debug configuration
    DEBUG: bool = False

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> "InstrumentationSettings":
        """
        Create settings from YAML config dict.

        Args:
            config: The host application's instrumentation configuration.

        Returns:
            InstrumentationSettings instance
        """
        kwargs = {}

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
        if "console_spans" in config:
            kwargs["CONSOLE_SPANS"] = config["console_spans"]

        settings = cls(**kwargs)

        # Auto-enable console spans when using console exporter
        if settings.EXPORTER == ExporterType.CONSOLE:
            settings.CONSOLE_SPANS = True

        return settings


# Singleton instance configured by the host application.
_settings: Optional[InstrumentationSettings] = None


def get_settings() -> InstrumentationSettings:
    """
    Get the configured settings instance.

    Raises:
        RuntimeError: If settings have not been configured.
    """
    if _settings is None:
        raise RuntimeError(
            "Instrumentation settings not configured. "
            "Call configure() before using the tracker."
        )
    return _settings


def configure(config: Dict[str, Any]) -> InstrumentationSettings:
    """
    Configure settings from a host-provided mapping.

    Args:
        config: Instrumentation configuration values.

    Returns:
        The configured InstrumentationSettings instance
    """
    global _settings
    _settings = InstrumentationSettings.from_config(config)
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
