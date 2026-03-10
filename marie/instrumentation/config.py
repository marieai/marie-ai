"""
Instrumentation Configuration - Unified config from YAML.

Configuration is ONLY loaded from YAML config via configure_from_yaml().
There is no fallback to environment variables - YAML is the single source of truth.
"""

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

    Must be initialized from YAML config via configure_from_yaml().
    There is no fallback - config must be explicitly provided.
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
        storage_config: Optional[Dict[str, Any]] = None,
    ) -> "InstrumentationSettings":
        """
        Create settings from YAML config dict.

        Args:
            config: The llm_tracking section from YAML config
            storage_config: Ignored (kept for backward compatibility)

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


# Singleton instance - must be configured via configure_from_yaml()
_settings: Optional[InstrumentationSettings] = None

# Backward compatibility alias
LLMTrackingSettings = InstrumentationSettings


def get_settings() -> InstrumentationSettings:
    """
    Get the configured settings instance.

    Raises:
        RuntimeError: If settings have not been configured via configure_from_yaml()
    """
    if _settings is None:
        raise RuntimeError(
            "Instrumentation settings not configured. "
            "Call configure_from_yaml() first or ensure llm_tracking section "
            "is present in the YAML config."
        )
    return _settings


def configure_from_yaml(
    config: Dict[str, Any],
    storage_config: Optional[Dict[str, Any]] = None,
) -> InstrumentationSettings:
    """
    Configure settings from YAML config (required).

    This is the ONLY way to initialize settings - there is no fallback.

    Args:
        config: The llm_tracking section from YAML config
        storage_config: Ignored (kept for backward compatibility)

    Returns:
        The configured InstrumentationSettings instance
    """
    global _settings
    _settings = InstrumentationSettings.from_config(config, storage_config)
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
