import importlib
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from marie.excepts import BadConfigSource
from marie.extract.validator.base import BaseValidator, ValidationStage
from marie.logging_core.predefined import default_logger as logger
from marie.wheel_manager import PipWheelManager, WheelDirectoryWatcher

from .builder_coercer import coerce_builder_fn
from .builder_types import TBuilder, TemplateBuilderFn
from .import_utils import import_submodules
from .parser_coercer import coerce_parser_fn
from .parser_types import ParserFn, TParser
from .region_processor_coercer import coerce_region_processor_fn
from .region_processor_types import RegionProcessorFn, TRegionProcessor
from .validator_coercer import coerce_validator_instance
from .validator_types import TValidator
from .wheel_callback import RegistryWheelCallback


class ComponentRegistry:
    """Thin facade that composes parsers, validators, and template builders."""

    def __init__(self):
        self._parsers: Dict[str, ParserFn] = {}
        self._validators: Dict[str, BaseValidator] = {}
        self._template_builders: Dict[str, TemplateBuilderFn] = {}
        self._region_processors: Dict[str, RegionProcessorFn] = (
            {}
        )  # New registry for region processors
        self._core_initialized = False
        self._external_modules_loaded = False
        self._auto_load_core = True
        self._lock = threading.RLock()

        self._wheel_manager = PipWheelManager()
        self._wheel_watcher = WheelDirectoryWatcher(self._wheel_manager)
        self._wheel_callback = RegistryWheelCallback(self)
        self._wheel_manager.add_callback(self._wheel_callback)

    def register_parser(self, name: str) -> Callable[[TParser], TParser]:
        def decorator(obj: TParser) -> TParser:
            with self._lock:
                if name in self._parsers:
                    logger.warning(f"Parser '{name}' already registered; overwriting.")
                self._parsers[name] = coerce_parser_fn(obj)  # type: ignore[arg-type]
            logger.info(f"Registered parser: {name} ({type(obj).__name__})")
            return obj

        return decorator

    def register_template_builder(self, name: str) -> Callable[[TBuilder], TBuilder]:
        def decorator(obj: TBuilder) -> TBuilder:
            with self._lock:
                if name in self._template_builders:
                    logger.warning(f"Template builder '{name}' already registered.")
                    raise BadConfigSource(f"Template builder {name} already registered")
                self._template_builders[name] = coerce_builder_fn(obj)  # type: ignore[arg-type]
            logger.info(f"Registered template_builder: {name} ({type(obj).__name__})")
            return obj

        return decorator

    def register_validator(self, name: str):
        def decorator(obj: TValidator):
            with self._lock:
                if name in self._validators:
                    logger.warning(
                        f"Validator '{name}' already registered; overwriting."
                    )
                    raise BadConfigSource(f"Validator '{name}' already registered")
                self._validators[name] = coerce_validator_instance(obj, name)
            logger.info(f"Registered validator: {name} ({type(obj).__name__})")
            return obj

        return decorator

    def register_region_processor(
        self, name: str
    ) -> Callable[[TRegionProcessor], TRegionProcessor]:
        """Register a region processor function or class."""

        def decorator(obj: TRegionProcessor) -> TRegionProcessor:
            with self._lock:
                if name in self._region_processors:
                    logger.warning(
                        f"Region processor '{name}' already registered; overwriting."
                    )
                self._region_processors[name] = coerce_region_processor_fn(obj)  # type: ignore[arg-type]
            logger.debug(f"Registered region processor: {name} ({type(obj).__name__})")
            return obj

        return decorator

    def register_validator_instance(self, validator: BaseValidator):
        with self._lock:
            self._validators[validator.name] = validator
        logger.debug(f"Registered validator instance: {validator.name}")

    def initialize_core_components(self):
        if not self._core_initialized:
            try:
                # import for side effects/registration
                import marie.extract.results.core.core_parsers  # noqa: F401
                import marie.extract.results.core.core_regions_processors  # noqa: F401
                import marie.extract.results.core.core_template_builders  # noqa: F401
                import marie.extract.results.core.core_validators  # noqa: F401

                self._core_initialized = True
                p, v, b, rp = self._counts()
                logger.info(
                    f"Initialized {p} core parsers, {v} core validators, {b} core template_builders, {rp} core regions processors"
                )
            except ImportError as e:
                logger.error(f"Failed to initialize core components: {e}")
                raise

    def initialize_external_components(
        self, component_modules: List[str], strict: bool = True
    ) -> Dict[str, object]:
        if self._external_modules_loaded:
            logger.debug("External components already loaded")
            p, v, b, rp = self._counts()
            return {
                "loaded": [],
                "failed": [],
                "total_parsers": p,
                "total_validators": v,
                "total_template_builders": b,
                "total_region_processors": rp,
            }

        importlib.invalidate_caches()
        all_loaded: List[str] = []
        all_failed: List[Tuple[str, str]] = []
        seen = set()

        for root in component_modules:
            loaded, failed = import_submodules(root, include_prefixes=None, seen=seen)
            all_loaded.extend(loaded)
            all_failed.extend(failed)
            if loaded:
                logger.info(f"Loaded {len(loaded)} modules from {root}")
            if failed:
                logger.warning(f"{len(failed)} failures while loading from {root}")
            if strict and failed:
                raise BadConfigSource(
                    f"Failed to load modules: {failed} while loading from {root}"
                )

        self._external_modules_loaded = True
        p, v, b, rp = self._counts()
        return {
            "loaded": all_loaded,
            "failed": all_failed,
            "total_parsers": p,
            "total_validators": v,
            "total_template_builders": b,
            "total_region_processors": rp,
        }

    def initialize_from_config(self, config: Dict[str, Any]):
        self._auto_load_core = config.get("load_core_components", True)
        if self._auto_load_core:
            self.initialize_core_components()

        p, v, b, rp = self._counts()
        result = {
            "loaded": [],
            "failed": [],
            "total_parsers": p,
            "total_validators": v,
            "total_template_builders": b,
            "total_region_processors": rp,
        }

        # wheels
        wheel_directories = config.get("wheel_directories", [])
        wheel_watch = config.get("watch_wheels", True)

        wheel_results_all = []
        for wheel_dir in wheel_directories:
            try:
                wheel_result = self._wheel_watcher.install_existing_wheels(wheel_dir)
                wheel_results_all.append({wheel_dir: wheel_result})
                if wheel_watch:
                    self._wheel_watcher.watch_directory(wheel_dir)
            except Exception as e:
                logger.error(f"Failed to handle wheels from {wheel_dir}: {e}")

        if wheel_results_all:
            result["wheel_results"] = wheel_results_all

        external_modules = config.get("external_component_modules", [])
        if external_modules:
            result = self.initialize_external_components(external_modules)
        return result

    def __init_core_components(self):
        if not self._core_initialized and self._auto_load_core:
            self.initialize_core_components()

    def get_parser(self, name: str) -> Optional[ParserFn]:
        self.__init_core_components()
        with self._lock:
            parser = self._parsers.get(name)
        if parser is None:
            available = list(self.list_parsers()) or "none"
            logger.warning(f"Parser '{name}' not found. Available: {available}")
        return parser

    def get_validator(self, name: str) -> Optional[BaseValidator]:
        self.__init_core_components()
        with self._lock:
            validator = self._validators.get(name)
        if validator is None:
            available = list(self.list_validators()) or "none"
            logger.warning(f"Validator '{name}' not found. Available: {available}")
        return validator

    def get_template_builder(self, name: str) -> Optional[TemplateBuilderFn]:
        self.__init_core_components()
        with self._lock:
            return self._template_builders.get(name)

    def get_region_processor(self, name: str) -> Optional[RegionProcessorFn]:
        """Get a region processor by name."""
        self.__init_core_components()
        with self._lock:
            processor = self._region_processors.get(name)
        if processor is None:
            available = list(self.list_region_processors()) or "none"
            logger.warning(
                f"Region processor '{name}' not found. Available: {available}"
            )
        return processor

    def list_parsers(self) -> List[str]:
        self.__init_core_components()
        with self._lock:
            return list(self._parsers.keys())

    def list_validators(self) -> List[str]:
        self.__init_core_components()
        with self._lock:
            return list(self._validators.keys())

    def list_template_builders(self) -> List[str]:
        self.__init_core_components()
        with self._lock:
            return list(self._template_builders.keys())

    def list_region_processors(self) -> List[str]:
        """List all registered region processor names."""
        self.__init_core_components()
        with self._lock:
            return list(self._region_processors.keys())

    def list_validators_for_stage(self, stage: ValidationStage) -> List[str]:
        self.__init_core_components()
        with self._lock:
            return [
                name
                for name, validator in self._validators.items()
                if validator.supports_stage(stage)
            ]

    def validators(self):
        self.__init_core_components()
        with self._lock:
            return dict(self._validators)

    def get_registry_info(self) -> Dict[str, Any]:
        self.__init_core_components()
        installed_wheels = self._wheel_manager.get_installed_wheels()
        p, v, b, rp = self._counts()
        info = {
            "total_parsers": p,
            "total_validators": v,
            "total_template_builders": b,
            "total_region_processors": rp,
            "parser_names": self.list_parsers(),
            "validator_names": self.list_validators(),
            "template_builder_names": self.list_template_builders(),
            "region_processor_names": self.list_region_processors(),
            "core_initialized": self._core_initialized,
            "external_loaded": self._external_modules_loaded,
            "auto_load_core": self._auto_load_core,
            "installed_wheels": {
                name: {
                    "package_name": data["package_name"],
                    "modules_count": len(data["modules"]),
                    "install_time": data["install_time"],
                }
                for name, data in installed_wheels.items()
            },
            "watched_directories": list(self._wheel_watcher.watched_directories),
        }
        return info

    def _counts(self) -> Tuple[int, int, int, int]:
        with self._lock:
            return (
                len(self._parsers),
                len(self._validators),
                len(self._template_builders),
                len(self._region_processors),
            )

    def get_wheel_manager(self) -> PipWheelManager:
        return self._wheel_manager

    def get_wheel_watcher(self) -> WheelDirectoryWatcher:
        return self._wheel_watcher

    def cleanup(self):
        self._wheel_watcher.stop_watching()
        self._wheel_manager.cleanup()
        logger.info("Component registry cleanup completed")


component_registry = ComponentRegistry()
register_parser = component_registry.register_parser
register_validator = component_registry.register_validator
register_template_builder = component_registry.register_template_builder
register_region_processor = component_registry.register_region_processor
