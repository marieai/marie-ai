import importlib
from typing import Any, Callable, Dict, List, Optional, Union

from marie.logging_core.predefined import default_logger as logger
from marie.wheel_manager import PipWheelManager, WheelDirectoryWatcher

from .base_validator import BaseValidator, FunctionValidatorWrapper, ValidationStage


class RegistryWheelCallback:
    """Callback handler for wheel installation events in the registry"""

    def __init__(self, registry):
        self.registry = registry

    def on_wheel_installed(self, wheel_info: Dict[str, Any]) -> None:
        """Called when a wheel is successfully installed"""
        parsers_count = len(self.registry._parsers)
        validators_count = len(self.registry._validators)
        logger.info(f"Wheel installed: {wheel_info['package_name']}")
        logger.info(
            f"Total components available: {parsers_count} parsers, {validators_count} validators"
        )

    def on_wheel_uninstalled(self, package_name: str) -> None:
        """Called when a wheel is successfully uninstalled"""
        parsers_count = len(self.registry._parsers)
        validators_count = len(self.registry._validators)
        logger.info(f"Wheel uninstalled: {package_name}")
        logger.info(
            f"Total components available: {parsers_count} parsers, {validators_count} validators"
        )

    def on_wheel_error(self, wheel_path: str, error: Exception) -> None:
        """Called when a wheel installation/uninstallation fails"""
        logger.error(f"Wheel installation failed for {wheel_path}: {error}")


class ComponentRegistry:

    def __init__(self):
        self._parsers: Dict[str, Callable] = {}
        self._validators: Dict[str, BaseValidator] = {}
        self._core_initialized = False
        self._external_modules_loaded = False
        self._auto_load_core = True

        # Wheel management components
        self._wheel_manager = PipWheelManager()
        self._wheel_watcher = WheelDirectoryWatcher(self._wheel_manager)
        self._wheel_callback = RegistryWheelCallback(self)
        self._wheel_manager.add_callback(self._wheel_callback)

    def register_parser(self, name: str):
        """Decorator to register a parser function"""

        def decorator(func: Callable):
            self._parsers[name] = func
            logger.debug(f"Registered parser: {name}")
            return func

        return decorator

    def register_validator(self, name: str):
        """Decorator to register a validator (BaseValidator instance, class, or function)"""

        def decorator(validator: Union[type, BaseValidator, Callable]):
            if isinstance(validator, type) and issubclass(validator, BaseValidator):
                validator_instance = validator()
                validator_instance.name = name
                self._validators[name] = validator_instance
            elif isinstance(validator, BaseValidator):
                validator.name = name
                self._validators[name] = validator
            elif callable(validator):
                wrapped_validator = FunctionValidatorWrapper(name, validator)
                self._validators[name] = wrapped_validator
            else:
                raise ValueError(
                    f"Invalid validator type: {type(validator)}. Must be BaseValidator class/instance or callable."
                )

            logger.debug(f"Registered validator: {name}")
            return validator

        return decorator

    def register_validator_instance(self, validator: BaseValidator):
        """Register a validator instance directly"""
        self._validators[validator.name] = validator
        logger.debug(f"Registered validator instance: {validator.name}")

    def initialize_core_components(self):
        """Initialize core components (parsers, validators) if not already done"""
        if not self._core_initialized:
            try:
                from . import core_parsers, core_validators

                self._core_initialized = True
                logger.info(
                    f"Initialized {len(self._parsers)} core parsers and {len(self._validators)} core validators"
                )
            except ImportError as e:
                logger.error(f"Failed to initialize core components: {e}")
                raise

    def initialize_external_components(self, component_modules: List[str]):
        """Initialize external components from configuration"""
        if self._external_modules_loaded:
            logger.debug("External components already loaded")
            return {
                'loaded': [],
                'failed': [],
                'total_parsers': len(self._parsers),
                'total_validators': len(self._validators),
            }

        loaded_modules = []
        failed_modules = []

        for module_name in component_modules:
            try:
                importlib.import_module(module_name)
                loaded_modules.append(module_name)
                logger.info(f"Loaded external components from: {module_name}")
            except ImportError as e:
                failed_modules.append((module_name, str(e)))
                logger.warning(
                    f"Failed to load external component module {module_name}: {e}"
                )
            except Exception as e:
                failed_modules.append((module_name, str(e)))
                logger.error(
                    f"Error loading external component module {module_name}: {e}"
                )

        self._external_modules_loaded = True

        if loaded_modules:
            logger.info(
                f"Successfully loaded {len(loaded_modules)} external component modules"
            )
        if failed_modules:
            logger.warning(
                f"Failed to load {len(failed_modules)} external component modules"
            )

        return {
            'loaded': loaded_modules,
            'failed': failed_modules,
            'total_parsers': len(self._parsers),
            'total_validators': len(self._validators),
        }

    def initialize_from_config(self, config: Dict[str, Any]):
        """Initialize components from configuration dictionary"""
        self._auto_load_core = config.get('load_core_components', True)

        if self._auto_load_core:
            self.initialize_core_components()

        result = {
            'loaded': [],
            'failed': [],
            'total_parsers': len(self._parsers),
            'total_validators': len(self._validators),
        }

        wheel_directories = config.get('wheel_directories', [])
        wheel_watch = config.get('watch_wheels', True)

        for wheel_dir in wheel_directories:
            try:
                wheel_result = self._wheel_watcher.install_existing_wheels(wheel_dir)
                result['wheel_results'] = wheel_result

                if wheel_watch:
                    self._wheel_watcher.watch_directory(wheel_dir)

            except Exception as e:
                logger.error(f"Failed to handle wheels from {wheel_dir}: {e}")

        external_modules = config.get('external_component_modules', [])
        if external_modules:
            result = self.initialize_external_components(external_modules)

        return result

    def __init_core_components(self):
        if not self._core_initialized and self._auto_load_core:
            self.initialize_core_components()

    def get_parser(self, name: str) -> Optional[Callable]:
        """Get a parser by name, auto-initializing core components if needed"""
        self.__init_core_components()
        parser = self._parsers.get(name)
        if parser is None:
            available = list(self._parsers.keys()) if self._parsers else "none"
            logger.warning(f"Parser '{name}' not found. Available: {available}")
        return parser

    def get_validator(self, name: str) -> Optional[BaseValidator]:
        """Get a validator by name, auto-initializing core components if needed"""
        self.__init_core_components()

        validator = self._validators.get(name)
        if validator is None:
            available = list(self._validators.keys()) if self._validators else "none"
            logger.warning(f"Validator '{name}' not found. Available: {available}")
        return validator

    def list_parsers(self) -> List[str]:
        """List all registered parser names"""
        self.__init_core_components()
        return list(self._parsers.keys())

    def list_validators(self) -> List[str]:
        """List all registered validator names"""
        self.__init_core_components()
        return list(self._validators.keys())

    def list_validators_for_stage(self, stage: ValidationStage) -> List[str]:
        """List validators that support a specific validation stage"""
        self.__init_core_components()

        return [
            name
            for name, validator in self._validators.items()
            if validator.supports_stage(stage)
        ]

    def validators(self):
        """Get all registered validators"""
        self.__init_core_components()
        return self._validators

    def get_validation_info(self) -> Dict[str, Any]:
        """Get detailed validation configuration information"""
        self.__init_core_components()

        validator_info = {}
        for name, validator in self._validators.items():
            validator_info[name] = {
                'name': validator.name,
                'supported_stages': [
                    stage.value for stage in validator.supported_stages
                ],
                'type': type(validator).__name__,
            }

        return {
            'total_validators': len(self._validators),
            'validator_details': validator_info,
            'available_stages': [stage.value for stage in ValidationStage],
        }

    def get_registry_info(self) -> Dict[str, Any]:
        """Get detailed information about the registry"""
        self.__init_core_components()

        installed_wheels = self._wheel_manager.get_installed_wheels()

        info = {
            'total_parsers': len(self._parsers),
            'total_validators': len(self._validators),
            'parser_names': list(self._parsers.keys()),
            'validator_names': list(self._validators.keys()),
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

        info['validation'] = self.get_validation_info()
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
        logger.info("Component registry cleanup completed")


# Global registry instance
component_registry = ComponentRegistry()
register_parser = component_registry.register_parser
register_validator = component_registry.register_validator
