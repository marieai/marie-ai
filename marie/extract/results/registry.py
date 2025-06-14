import importlib
from typing import Any, Callable, Dict, List, Optional

from marie.logging_core.predefined import default_logger as logger
from marie.wheel_manager import (
    PipWheelManager,
    WheelDirectoryWatcher,
    WheelInstallationCallback,
)


class ParserRegistryWheelCallback:
    """Callback handler for wheel installation events in ParserRegistry"""

    def __init__(self, parser_registry):
        self.parser_registry = parser_registry

    def on_wheel_installed(self, wheel_info: Dict[str, Any]) -> None:
        """Called when a wheel is successfully installed"""
        parsers_count = len(self.parser_registry._parsers)
        logger.info(f"Wheel installed: {wheel_info['package_name']}")
        logger.info(f"Total parsers available: {parsers_count}")

    def on_wheel_uninstalled(self, package_name: str) -> None:
        """Called when a wheel is successfully uninstalled"""
        parsers_count = len(self.parser_registry._parsers)
        logger.info(f"Wheel uninstalled: {package_name}")
        logger.info(f"Total parsers available: {parsers_count}")

    def on_wheel_error(self, wheel_path: str, error: Exception) -> None:
        """Called when a wheel installation/uninstallation fails"""
        logger.error(f"Wheel installation failed for {wheel_path}: {error}")


class ParserRegistry:
    def __init__(self):
        self._parsers: Dict[str, Callable] = {}
        self._core_initialized = False
        self._external_modules_loaded = False
        self._auto_load_core = True
        # Wheel management components
        self._wheel_manager = PipWheelManager()
        self._wheel_watcher = WheelDirectoryWatcher(self._wheel_manager)
        self._wheel_callback = ParserRegistryWheelCallback(self)
        self._wheel_manager.add_callback(self._wheel_callback)

    def register(self, name: str):
        """Decorator to register a parser function"""

        def decorator(func: Callable):
            self._parsers[name] = func
            logger.debug(f"Registered parser: {name}")
            return func

        return decorator

    def initialize_core_parsers(self):
        """Initialize core parsers if not already done"""
        if not self._core_initialized:
            try:
                from . import core_parsers

                self._core_initialized = True
                logger.info(
                    f"Initialized {len(self._parsers)} core parsers: {list(self._parsers.keys())}"
                )
            except ImportError as e:
                logger.error(f"Failed to initialize core parsers: {e}")
                raise

    def initialize_external_parsers(self, parser_modules: List[str]):
        """Initialize external parsers from configuration"""
        if self._external_modules_loaded:
            logger.debug("External parsers already loaded")
            return

        loaded_modules = []
        failed_modules = []

        for module_name in parser_modules:
            try:
                importlib.import_module(module_name)
                loaded_modules.append(module_name)
                logger.info(f"Loaded external parsers from: {module_name}")
            except ImportError as e:
                failed_modules.append((module_name, str(e)))
                logger.warning(
                    f"Failed to load external parser module {module_name}: {e}"
                )
            except Exception as e:
                failed_modules.append((module_name, str(e)))
                logger.error(f"Error loading external parser module {module_name}: {e}")

        self._external_modules_loaded = True

        if loaded_modules:
            logger.info(
                f"Successfully loaded {len(loaded_modules)} external parser modules"
            )
        if failed_modules:
            logger.warning(
                f"Failed to load {len(failed_modules)} external parser modules"
            )

        return {
            'loaded': loaded_modules,
            'failed': failed_modules,
            'total_parsers': len(self._parsers),
        }

    def initialize_from_config(self, config: Dict[str, Any]):
        """Initialize parsers from configuration dictionary"""
        self._auto_load_core = config.get('load_core_parsers', True)

        if self._auto_load_core:
            self.initialize_core_parsers()

        result = {'loaded': [], 'failed': [], 'total_parsers': len(self._parsers)}

        # Handle wheel directories
        wheel_directories = config.get('wheel_directories', [])
        wheel_watch = config.get('watch_wheels', True)

        for wheel_dir in wheel_directories:
            try:
                wheel_result = self._wheel_watcher.install_existing_wheels(wheel_dir)
                result['wheel_results'] = wheel_result

                # Start watching if requested
                if wheel_watch:
                    self._wheel_watcher.watch_directory(wheel_dir)

            except Exception as e:
                logger.error(f"Failed to handle wheels from {wheel_dir}: {e}")

        external_modules = config.get('external_parser_modules', [])
        if external_modules:
            result = self.initialize_external_parsers(external_modules)

        return result

    def get_parser(self, name: str) -> Optional[Callable]:
        """Get a parser by name, only auto-initializing core parsers if allowed"""
        if not self._core_initialized and self._auto_load_core:
            self.initialize_core_parsers()

        parser = self._parsers.get(name)
        if parser is None:
            available_parsers = list(self._parsers.keys()) if self._parsers else "none"
            logger.warning(
                f"Parser '{name}' not found. Available parsers: {available_parsers}"
            )
        return parser

    def list_parsers(self) -> List[str]:
        """List all registered parser names"""
        if not self._core_initialized and self._auto_load_core:
            self.initialize_core_parsers()
        return list(self._parsers.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a parser is registered"""
        if not self._core_initialized and self._auto_load_core:
            self.initialize_core_parsers()
        return name in self._parsers

    def get_parser_info(self) -> Dict[str, Any]:
        """Get detailed information about registered parsers"""
        if not self._core_initialized and self._auto_load_core:
            self.initialize_core_parsers()

        installed_wheels = self._wheel_manager.get_installed_wheels()

        info = {
            'total_parsers': len(self._parsers),
            'parser_names': list(self._parsers.keys()),
            'core_initialized': self._core_initialized,
            'external_loaded': self._external_modules_loaded,
            'auto_load_core': self._auto_load_core,
            'installed_wheels': {
                name: {
                    'package_name': data['package_name'],
                    'modules_count': len(data['modules']),
                    'install_time': data['install_time'],
                }
                for name, data in installed_wheels.items()
            },
            'watched_directories': list(self._wheel_watcher.watched_directories),
        }

        # Add parser documentation if available
        parser_docs = {}
        for name, parser_func in self._parsers.items():
            if hasattr(parser_func, '__doc__') and parser_func.__doc__:
                parser_docs[name] = parser_func.__doc__.strip().split('\n')[0]

        if parser_docs:
            info['parser_descriptions'] = parser_docs

        return info

    def get_wheel_manager(self) -> PipWheelManager:
        """Get the wheel manager for advanced operations"""
        return self._wheel_manager

    def get_wheel_watcher(self) -> WheelDirectoryWatcher:
        """Get the wheel watcher for advanced operations"""
        return self._wheel_watcher

    def cleanup(self):
        """Clean up all resources"""
        self._wheel_watcher.stop_watching()
        self._wheel_manager.cleanup()
        logger.info("Parser registry cleanup completed")


# Global registry instance
parser_registry = ParserRegistry()
register_parser = parser_registry.register
