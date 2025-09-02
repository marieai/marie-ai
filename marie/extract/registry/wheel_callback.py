from typing import TYPE_CHECKING, Any, Dict

from marie.logging_core.predefined import default_logger as logger

# forward-ref
if TYPE_CHECKING:
    from .base import ComponentRegistry


class RegistryWheelCallback:
    """Callback handler for wheel installation events in the registry"""

    def __init__(self, registry: "ComponentRegistry"):
        self.registry = registry

    def on_wheel_installed(self, wheel_info: Dict[str, Any]) -> None:
        p = len(self.registry._parsers)
        v = len(self.registry._validators)
        b = len(self.registry._template_builders)
        logger.info(f"Wheel installed: {wheel_info.get('package_name')}")
        logger.info(
            f"Total components available: {p} parsers, {v} validators, {b} template_builders"
        )

    def on_wheel_uninstalled(self, package_name: str) -> None:
        p = len(self.registry._parsers)
        v = len(self.registry._validators)
        b = len(self.registry._template_builders)
        logger.info(f"Wheel uninstalled: {package_name}")
        logger.info(
            f"Total components available: {p} parsers, {v} validators, {b} template_builders"
        )

    def on_wheel_error(self, wheel_path: str, error: Exception) -> None:
        logger.error(f"Wheel installation failed for {wheel_path}: {error}")
