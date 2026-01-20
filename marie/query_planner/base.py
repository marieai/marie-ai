import enum
import importlib
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field, field_validator

from marie.logging_core.predefined import default_logger as logger
from marie.query_planner.model import QueryPlannersConf
from marie.wheel_manager import PipWheelManager, WheelDirectoryWatcher

DEFAULT_NAME = "query_plan_tool"

QUERYNODE_QUERY_STR_DESC = """\
Question we are asking. This is the query string that will be executed. \
"""

QUERYNODE_TOOL_NAME_DESC = """\
Name of the tool to execute the `query_str`. \
Should NOT be specified if there are subquestions to be specified, in which \
case child_nodes should be nonempty instead.\
"""

QUERYNODE_DEPENDENCIES_DESC = """\
List of sub-questions that need to be answered in order \
to answer the question given by `query_str`.\
Should be blank if there are no sub-questions to be specified, in which case \
`tool_name` is specified.\
"""

QUERYNODE_TYPE_DESC = """\
Type of question we are asking, either a single question or a multi question merge when there are multiple questions.
"""


class PlannerMetadata(BaseModel):
    """Metadata for a registered query planner"""

    planner_id: str = Field(
        ..., description="Unique identifier for the planner (same as name)"
    )
    name: str = Field(..., description="Unique name of the planner")
    description: Optional[str] = Field(None, description="Description of the planner")
    version: str = Field(default="1.0.0", description="Version of the planner")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    category: Optional[str] = Field(None, description="Category of the planner")
    source_type: str = Field(..., description="Source type: 'code', 'json', 'wheel'")
    source_module: Optional[str] = Field(
        None, description="Python module path (for code-based planners)"
    )
    plan_definition: Optional[Dict[str, Any]] = Field(
        None, description="JSON plan definition (for JSON-based planners)"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Creation timestamp",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Last update timestamp",
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class QueryPlanRegistryWheelCallback:
    """Callback handler for wheel installation events in QueryPlanRegistry"""

    @staticmethod
    def on_wheel_installed(wheel_info: Dict[str, Any]) -> None:
        """Called when a wheel is successfully installed"""
        planners_count = len(QueryPlanRegistry.list_planners())
        logger.info(f"Wheel installed: {wheel_info['package_name']}")
        logger.info(f"Total query planners available: {planners_count}")

    @staticmethod
    def on_wheel_uninstalled(package_name: str) -> None:
        """Called when a wheel is successfully uninstalled"""
        planners_count = len(QueryPlanRegistry.list_planners())
        logger.info(f"Wheel uninstalled: {package_name}")
        logger.info(f"Total query planners available: {planners_count}")

    @staticmethod
    def on_wheel_error(wheel_path: str, error: Exception) -> None:
        """Called when a wheel installation/uninstallation fails"""
        logger.error(f"Wheel installation failed for {wheel_path}: {error}")


class QueryPlanRegistry:
    """Registry for query planner functions with wheel support and JSON-based planners."""

    _plans: Dict[str, Callable] = {}
    _metadata: Dict[str, PlannerMetadata] = (
        {}
    )  # Store metadata for each planner (keyed by name)
    _id_to_name: Dict[str, str] = {}  # Mapping from planner_id to name
    _external_modules_loaded: bool = False
    _storage_path: Optional[Path] = None  # Path to store JSON planners

    # Wheel management components (class-level)
    _wheel_manager: Optional[PipWheelManager] = None
    _wheel_watcher: Optional[WheelDirectoryWatcher] = None
    _wheel_callback: Optional[QueryPlanRegistryWheelCallback] = None

    @classmethod
    def _ensure_wheel_manager(cls):
        """Ensure wheel management components are initialized"""
        if cls._wheel_manager is None:
            cls._wheel_manager = PipWheelManager()
            cls._wheel_watcher = WheelDirectoryWatcher(cls._wheel_manager)
            cls._wheel_callback = QueryPlanRegistryWheelCallback()
            cls._wheel_manager.add_callback(cls._wheel_callback)

    @classmethod
    def register(cls, name: str, function: Callable = None):
        """
        Register a query planner function.

        Usage:
            As a decorator:
                @QueryPlanRegistry.register("my_planner")
                def my_query_planner(planner_info):
                    # planner implementation
                    return plan

            Direct registration:
                def my_query_planner(planner_info):
                    # planner implementation
                    return plan
                QueryPlanRegistry.register("my_planner", my_query_planner)

        :param name: The name to register the planner under. If None, uses the function name.
        :param function: Optional. The function to register directly.
        :return: Decorator function if no function is provided; otherwise, None.
        """

        logger.info(f"Registering query planner function : {name} from {function}")

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            planner_name = name or func.__name__

            if planner_name in cls._plans:
                raise ValueError(
                    f"Query planner '{planner_name}' is already registered"
                )

            cls._plans[planner_name] = wrapper

            # Create metadata for code-based planners
            module_name = func.__module__ if hasattr(func, '__module__') else None
            description = func.__doc__.strip() if func.__doc__ else None

            # Try to extract plan definition by calling the planner with dummy PlannerInfo
            plan_definition = None
            try:
                from marie.job.job_manager import generate_job_id
                from marie.query_planner.base import PlannerInfo

                # Create dummy PlannerInfo to extract plan structure
                # Use proper UUID7 format for base_id (required by increment_uuid7str)
                dummy_info = PlannerInfo(
                    name=planner_name,
                    base_id=generate_job_id(),
                    current_id=0,
                    steps=[],
                    metadata=None,
                )

                # Call the planner to get its QueryPlan
                query_plan = wrapper(dummy_info)

                # Convert QueryPlan to dict for storage
                if hasattr(query_plan, 'model_dump'):
                    plan_definition = query_plan.model_dump()
                elif hasattr(query_plan, 'dict'):
                    plan_definition = query_plan.dict()

                logger.debug(
                    f"Extracted plan definition for '{planner_name}' with {len(plan_definition.get('nodes', []))} nodes"
                )
            except Exception as e:
                logger.warning(
                    f"Could not extract plan definition for '{planner_name}': {e}"
                )
                plan_definition = None

            metadata = PlannerMetadata(
                planner_id=planner_name,  # Use name as ID
                name=planner_name,
                description=description,
                version="1.0.0",
                tags=[],
                category=None,
                source_type="code",
                source_module=module_name,
                plan_definition=plan_definition,
            )
            cls._metadata[planner_name] = metadata
            cls._id_to_name[planner_name] = planner_name  # ID = name

            logger.info(
                f"Registered planner '{planner_name}' (ID: {planner_name}, has_definition: {plan_definition is not None})"
            )
            return wrapper

        if function is not None:
            return decorator(function)

        return decorator

    @classmethod
    def register_from_module(cls, planner_module: str) -> bool:
        """
        Registers a planner from the specified module.

        :param planner_module: The name of the module to register the planner from.
        :type planner_module: str
        :return: True if successful, False if failed
        """
        try:
            logger.info(f"Registering planner from {planner_module}")
            importlib.import_module(planner_module)
            logger.info(f"Successfully registered planner from {planner_module}")
            return True
        except Exception as e:
            logger.error(f"Error registering planner from {planner_module}: {e}")
            warnings.warn(
                f"Error importing {planner_module} : some configs may not be available\n\n\tRoot cause: {e}\n"
            )
            return False

    @classmethod
    def discover_from_package(
        cls,
        package_name: str,
        pattern: str = "*",
    ) -> Dict[str, Any]:
        """
        Auto-discover and register planners from a Python package.

        Scans the package directory for subdirectories matching the given pattern
        and imports any .py files that contain the @register_query_plan decorator.

        :param package_name: The fully qualified package name (e.g., 'grapnel_g5.extract.providers')
        :param pattern: Glob pattern for matching subdirectory names (default: '*')
        :return: Dictionary with 'loaded', 'failed', and 'skipped' lists
        """
        from fnmatch import fnmatch
        from pathlib import Path

        result: Dict[str, Any] = {'loaded': [], 'failed': [], 'skipped': []}

        logger.info(
            f"Discovering planners from package '{package_name}' (pattern: {pattern})"
        )

        try:
            # Import the base package to get its path
            pkg = importlib.import_module(package_name)
            if not hasattr(pkg, '__file__') or pkg.__file__ is None:
                logger.warning(f"Package '{package_name}' has no __file__ attribute")
                result['error'] = (
                    f"Package '{package_name}' is a namespace package without __file__"
                )
                return result

            pkg_path = Path(pkg.__file__).parent

            # Scan for subdirectories matching the pattern
            discovered_count = 0
            for item in sorted(pkg_path.iterdir()):
                # Skip non-directories and private/dunder directories
                if not item.is_dir() or item.name.startswith('_'):
                    continue

                # Check if directory matches the pattern
                if not fnmatch(item.name, pattern):
                    result['skipped'].append(item.name)
                    continue

                # Scan all .py files in the directory for @register_query_plan
                found_planner = False
                for py_file in item.glob("*.py"):
                    if py_file.name.startswith('_'):
                        continue

                    # Check if file contains the decorator
                    try:
                        content = py_file.read_text(encoding='utf-8')
                        if '@register_query_plan' not in content:
                            continue
                    except Exception as e:
                        logger.debug(f"Could not read {py_file}: {e}")
                        continue

                    # Build the full module path and try to import
                    module_name = py_file.stem
                    full_module_path = f"{package_name}.{item.name}.{module_name}"
                    if cls.register_from_module(full_module_path):
                        result['loaded'].append(full_module_path)
                        discovered_count += 1
                        found_planner = True
                    else:
                        result['failed'].append(full_module_path)

                if not found_planner:
                    result['skipped'].append(f"{item.name} (no @register_query_plan)")

            logger.info(
                f"Discovered {discovered_count} planners from '{package_name}' "
                f"(skipped: {len(result['skipped'])}, failed: {len(result['failed'])})"
            )

        except ImportError as e:
            logger.error(f"Failed to import package '{package_name}': {e}")
            result['error'] = str(e)
        except Exception as e:
            logger.error(f"Error discovering planners from '{package_name}': {e}")
            result['error'] = str(e)

        return result

    @classmethod
    def initialize_from_config(
        cls, query_planners_conf: QueryPlannersConf
    ) -> Dict[str, Any]:
        """
        Initialize query planners from configuration with wheel support.

        :param query_planners_conf: Configuration containing planner modules and wheel settings
        :return: Dictionary with initialization results
        """
        logger.info("Initializing query planners from configuration")

        cls._ensure_wheel_manager()

        result = {
            'loaded': [],
            'failed': [],
            'discovered': {},
            'total_planners': len(cls._plans),
            'wheel_results': {},
        }

        # Handle wheel directories
        wheel_directories = query_planners_conf.wheel_directories
        wheel_watch = query_planners_conf.watch_wheels

        if wheel_directories:
            logger.info(f"Processing wheel directories: {wheel_directories}")

            for wheel_dir in wheel_directories:
                try:
                    logger.info(f"Installing wheels from directory: {wheel_dir}")
                    wheel_result = cls._wheel_watcher.install_existing_wheels(wheel_dir)
                    result['wheel_results'][wheel_dir] = wheel_result
                    logger.info(
                        f"Wheel installation result for {wheel_dir}: {wheel_result}"
                    )

                    # Start watching if requested
                    if wheel_watch:
                        logger.info(
                            f"Starting wheel directory watcher for: {wheel_dir}"
                        )
                        cls._wheel_watcher.watch_directory(wheel_dir)

                except Exception as e:
                    logger.error(f"Failed to handle wheels from {wheel_dir}: {e}")
                    result['failed'].append((wheel_dir, f"Wheel error: {e}"))

        # Auto-discover planners from packages
        if query_planners_conf.discover_packages:
            for disc_pkg in query_planners_conf.discover_packages:
                disc_result = cls.discover_from_package(
                    package_name=disc_pkg.package,
                    pattern=disc_pkg.pattern,
                )
                result['discovered'][disc_pkg.package] = disc_result
                # Add discovered modules to the overall loaded/failed lists
                result['loaded'].extend(disc_result.get('loaded', []))
                if disc_result.get('failed'):
                    result['failed'].extend(
                        [(m, "Discovery import failed") for m in disc_result['failed']]
                    )

        # Register planners from external modules
        loaded_modules = []
        failed_modules = []

        for planner in query_planners_conf.planners:
            logger.info(f"Registering planner: {planner.name} from {planner.py_module}")
            planner_module = planner.py_module

            if cls.register_from_module(planner_module):
                loaded_modules.append(planner_module)
            else:
                failed_modules.append((planner_module, "Module import failed"))

        result['loaded'] = loaded_modules
        result['failed'].extend(failed_modules)
        result['total_planners'] = len(cls._plans)

        cls._external_modules_loaded = True

        if loaded_modules:
            logger.info(f"Successfully loaded {len(loaded_modules)} planner modules")
        if failed_modules:
            logger.warning(f"Failed to load {len(failed_modules)} planner modules")

        if loaded_modules and not failed_modules:
            logger.info(
                f"All {len(loaded_modules)} planner modules loaded successfully"
            )
        elif failed_modules and not loaded_modules:
            logger.warning(f"All {len(failed_modules)} planner modules failed to load")
        elif loaded_modules and failed_modules:
            logger.info(
                f"Mixed results: {len(loaded_modules)} loaded, {len(failed_modules)} failed"
            )

        return result

    @classmethod
    def get(cls, planner_name: str) -> Callable:
        """
        Get a query planner by name.

        :param planner_name: The name of the query planner to get.
        :return: The query planner function.
        :raises ValueError: If the query planner is not registered.
        """
        try:
            return cls._plans[planner_name]
        except KeyError as e:
            raise KeyError(
                f"Query planner '{planner_name}' is not registered! Available planners are: {', '.join(cls._plans.keys())}"
            ) from e

    @classmethod
    def list_planners(cls) -> list[str]:
        """Return a list of all registered planner names."""
        return list(cls._plans.keys())

    @classmethod
    def get_planner_info(cls) -> Dict[str, Any]:
        """Get detailed information about registered planners"""
        cls._ensure_wheel_manager()

        installed_wheels = cls._wheel_manager.get_installed_wheels()
        planners = cls.list_planners()

        info = {
            'total_planners': len(planners),
            'planner_names': planners,
            'external_loaded': cls._external_modules_loaded,
            'installed_wheels': {
                name: {
                    'package_name': data['package_name'],
                    'modules_count': len(data['modules']),
                    'install_time': data['install_time'],
                }
                for name, data in installed_wheels.items()
            },
            'watched_directories': (
                list(cls._wheel_watcher.watched_directories)
                if cls._wheel_watcher
                else []
            ),
        }

        return info

    @classmethod
    def get_wheel_manager(cls) -> Optional[PipWheelManager]:
        """Get the wheel manager for advanced operations"""
        cls._ensure_wheel_manager()
        return cls._wheel_manager

    @classmethod
    def get_wheel_watcher(cls) -> Optional[WheelDirectoryWatcher]:
        """Get the wheel watcher for advanced operations"""
        cls._ensure_wheel_manager()
        return cls._wheel_watcher

    @classmethod
    def set_storage_path(cls, path: str):
        """
        Set the storage path for JSON planners.

        :param path: Directory path to store JSON planner definitions
        """
        cls._storage_path = Path(path)
        cls._storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"JSON planner storage path set to: {cls._storage_path}")

    @classmethod
    def register_from_json(
        cls,
        name: str,
        plan_definition: Dict[str, Any],
        description: Optional[str] = None,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> bool:
        """
        Register a query planner from a JSON plan definition.

        This allows Marie Studio to publish query plan templates directly to the gateway
        without requiring Python code.

        :param name: Unique name for the planner
        :param plan_definition: JSON dict containing the QueryPlan structure
        :param description: Optional description of the planner
        :param version: Version of the planner (default: "1.0.0")
        :param tags: Optional list of tags for categorization
        :param category: Optional category for the planner
        :return: True if successful, False if failed
        """
        try:
            if name in cls._plans:
                raise ValueError(f"Planner '{name}' is already registered")

            # Validate plan definition by attempting to parse it
            from marie.query_planner.base import QueryPlan

            query_plan = QueryPlan(**plan_definition)

            # Create a function that returns this plan
            def json_planner_function(
                planner_info: 'PlannerInfo', **kwargs
            ) -> 'QueryPlan':
                """Dynamically created planner from JSON definition"""
                # Return a copy of the plan with updated task IDs based on planner_info
                return query_plan

            # Register the function
            cls._plans[name] = json_planner_function

            # Store metadata
            metadata = PlannerMetadata(
                planner_id=name,  # Use name as ID
                name=name,
                description=description,
                version=version,
                tags=tags or [],
                category=category,
                source_type="json",
                plan_definition=plan_definition,
            )
            cls._metadata[name] = metadata

            # Maintain ID to name mapping (both are same now)
            cls._id_to_name[name] = name

            # Persist to storage if path is set
            if cls._storage_path:
                cls._save_json_planner(name, metadata)

            logger.info(f"Successfully registered JSON planner: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register JSON planner '{name}': {e}")
            return False

    @classmethod
    def _save_json_planner(cls, name: str, metadata: PlannerMetadata):
        """Save JSON planner metadata to storage"""
        if not cls._storage_path:
            return

        planner_file = cls._storage_path / f"{name}.json"
        with open(planner_file, 'w') as f:
            json.dump(metadata.model_dump(), f, indent=2)
        logger.debug(f"Saved JSON planner to: {planner_file}")

    @classmethod
    def load_json_planners_from_storage(cls) -> Dict[str, Any]:
        """
        Load all JSON planners from the storage directory.

        :return: Dictionary with loaded/failed counts
        """
        if not cls._storage_path or not cls._storage_path.exists():
            logger.warning("JSON planner storage path not set or doesn't exist")
            return {'loaded': 0, 'failed': 0}

        result = {'loaded': 0, 'failed': 0, 'planners': []}

        for planner_file in cls._storage_path.glob("*.json"):
            try:
                with open(planner_file, 'r') as f:
                    metadata_dict = json.load(f)

                metadata = PlannerMetadata(**metadata_dict)

                # Register the planner
                if metadata.plan_definition:
                    success = cls.register_from_json(
                        name=metadata.name,
                        plan_definition=metadata.plan_definition,
                        description=metadata.description,
                        version=metadata.version,
                        tags=metadata.tags,
                        category=metadata.category,
                    )

                    if success:
                        result['loaded'] += 1
                        result['planners'].append(metadata.name)
                    else:
                        result['failed'] += 1

            except Exception as e:
                logger.error(f"Failed to load JSON planner from {planner_file}: {e}")
                result['failed'] += 1

        logger.info(
            f"Loaded {result['loaded']} JSON planners, {result['failed']} failed"
        )
        return result

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a planner by name.

        :param name: Name of the planner to unregister
        :return: True if successful, False if not found
        """
        if name not in cls._plans:
            logger.warning(f"Planner '{name}' not found in registry")
            return False

        # Remove from registry
        del cls._plans[name]

        # Remove metadata
        metadata = cls._metadata.pop(name, None)

        # Remove ID mapping
        if metadata and metadata.planner_id in cls._id_to_name:
            del cls._id_to_name[metadata.planner_id]

        # Remove from storage if it's a JSON planner
        if metadata and metadata.source_type == "json" and cls._storage_path:
            planner_file = cls._storage_path / f"{name}.json"
            if planner_file.exists():
                planner_file.unlink()
                logger.debug(f"Deleted JSON planner file: {planner_file}")

        logger.info(f"Unregistered planner: {name}")
        return True

    @classmethod
    def get_metadata(cls, name: str) -> Optional[PlannerMetadata]:
        """
        Get metadata for a specific planner by name.

        :param name: Name of the planner
        :return: PlannerMetadata if found, None otherwise
        """
        return cls._metadata.get(name)

    @classmethod
    def get_metadata_by_id(cls, planner_id: str) -> Optional[PlannerMetadata]:
        """
        Get metadata for a specific planner by ID.

        :param planner_id: ID of the planner
        :return: PlannerMetadata if found, None otherwise
        """
        name = cls._id_to_name.get(planner_id)
        if name:
            return cls._metadata.get(name)
        return None

    @classmethod
    def unregister_by_id(cls, planner_id: str) -> bool:
        """
        Unregister a planner by ID.

        :param planner_id: ID of the planner to unregister
        :return: True if successful, False if not found
        """
        name = cls._id_to_name.get(planner_id)
        if name:
            return cls.unregister(name)
        logger.warning(f"Planner with ID '{planner_id}' not found in registry")
        return False

    @classmethod
    def list_planners_with_metadata(cls) -> List[Dict[str, Any]]:
        """
        List all registered planners with their metadata.

        :return: List of dictionaries containing planner info
        """
        result = []

        for name in cls._plans.keys():
            metadata = cls._metadata.get(name)

            if metadata:
                result.append(metadata.model_dump())
            else:
                # Fallback for legacy planners without metadata (shouldn't happen anymore)
                logger.warning(
                    f"Planner '{name}' has no metadata - this shouldn't happen"
                )
                result.append(
                    {
                        'planner_id': None,
                        'name': name,
                        'description': None,
                        'version': '1.0.0',
                        'tags': [],
                        'category': None,
                        'source_type': 'code',
                        'source_module': None,
                        'plan_definition': None,
                        'created_at': None,
                        'updated_at': None,
                    }
                )

        return result

    @classmethod
    def cleanup(cls):
        """Clean up all resources"""
        if cls._wheel_watcher:
            cls._wheel_watcher.stop_watching()
        if cls._wheel_manager:
            cls._wheel_manager.cleanup()
        logger.info("Query plan registry cleanup completed")


def register_query_plan(name: str = None):
    """
    Decorator to register a query planner function.

    This is a more concise alternative to @QueryPlannerRegistry.register

    Usage:
        @register_query_plan("my_query_plan")
        def my_query_planner(planner_info):
            # planner implementation
            return plan

    :param name: The name to register the query plan under. If None, uses the function name.
    :return: Decorator function
    """
    return QueryPlanRegistry.register(name)


class QueryTypeRegistry:
    _method_to_class: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, method: str):
        """Decorator to register a class for a specific method"""

        def decorator(model_class: Type[BaseModel]):
            cls._method_to_class[method] = model_class
            return model_class

        return decorator

    @classmethod
    def get_class_for_method(cls, method: str) -> Optional[Type[BaseModel]]:
        """Get the class for a given method"""
        return cls._method_to_class.get(method)


class QueryDefinition(BaseModel):
    """
    Abstract base class for query definitions. Represents a task executed via a specific method.
    Type of execution method (LLM, Python, API, etc.). Input parameters required for execution.
    """

    method: str
    endpoint: str = Field(
        ...,
        description="API endpoint for the query. This could a executor endpoint or a model endpoint.",
    )
    params: dict

    def validate_params(self):
        """Validate the parameters for the query."""
        raise NotImplementedError("Subclasses must implement validate_params.")


@QueryTypeRegistry.register("NOOP")
class NoopQueryDefinition(QueryDefinition):
    """
    Represents a NOOP (no operation) query definition.
    It can be used to group tasks in a DAG. The task is evaluated by the scheduler but never processed by the executor.
    """

    method: str = "NOOP"
    endpoint: str = "noop"
    params: dict = Field(default_factory=lambda: {"layout": None})

    def validate_params(self):
        if "layout" not in self.params or self.params["layout"] is None:
            raise ValueError("NOOP queries must have a 'layout' parameter.")


@QueryTypeRegistry.register("LLM")
class LlmQueryDefinition(QueryDefinition):
    """
    Represents an LLM (language model) query definition.
    """

    method: str = "LLM"
    endpoint: str = "extract"
    model_name: str = Field(..., description="Name of the LLM model to use.")
    params: dict = Field(default_factory=lambda: {"layout": None})

    def validate_params(self):
        if "layout" not in self.params or self.params["layout"] is None:
            raise ValueError("LLM queries must have a 'layout' parameter.")


@QueryTypeRegistry.register("PYTHON_FUNCTION")
class PythonFunctionQueryDefinition(QueryDefinition):
    """
    Represents a Python function execution query definition.
    This is dynamic and can be used to execute any Python function that is registered.
    """

    method: str = "PYTHON_FUNCTION"
    params: dict = Field(default_factory=lambda: {"layout": None, "function": None})

    def validate_params(self):
        if "layout" not in self.params or self.params["layout"] is None:
            raise ValueError("Python function queries must have a 'layout' parameter.")
        if "function" not in self.params or not isinstance(
            self.params.get("function"), str
        ):
            raise ValueError(
                "Python function queries must specify the 'function' name as a string."
            )


@QueryTypeRegistry.register("EXECUTOR_ENDPOINT")
class ExecutorEndpointQueryDefinition(QueryDefinition):
    """
    Represents an executor endpoint query definition.
    This is dynamic and can be used to execute any endpoint that is registered.
    """

    method: str = "EXECUTOR_ENDPOINT"
    params: dict = Field(default_factory=lambda: {"layout": None, "function": None})

    def validate_params(self):
        pass


class QueryType(str, enum.Enum):
    """
    Enumeration representing the types of queries that can be asked to a question answer system.
    """

    EXTRACTOR = "EXTRACTOR"
    SEGMENTER = "SEGMENTER"
    COMPUTE = "COMPUTE"
    MERGER = "MERGER"
    BRANCH = "BRANCH"  # Conditional branching node
    SWITCH = "SWITCH"  # Multi-way switch node
    GUARDRAIL = "GUARDRAIL"  # Quality validation node


class ComputeQuery(BaseModel):
    """
    Models a computation of a query, assume this can be some RAG system like llamaindex
    """

    query_str: str
    response: str = "..."


class Query(BaseModel):
    """
    Class representing a single question in a question answer subquery.
    Can be either a single question or a multi question merge.
    """

    task_id: str = Field(..., description="Unique id of the query")
    query_str: str = Field(
        ...,
        description=QUERYNODE_QUERY_STR_DESC,
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description=QUERYNODE_DEPENDENCIES_DESC,
    )
    node_type: QueryType = Field(
        default=QueryType.COMPUTE,
        description=QUERYNODE_TYPE_DESC,
    )

    definition: Optional[Any] = Field(
        default=None,
        description="Definition of the query to be executed.",
    )

    @field_validator('definition', mode='before')
    @classmethod
    def validate_definition(cls, v):
        """Custom serialization for the definition field"""
        if isinstance(v, dict):
            method = v.get('method', '')
            model_class = QueryTypeRegistry.get_class_for_method(method)
            if model_class:
                return model_class(**v)
            raise ValueError(
                f"Unknown method {method}, cannot create QueryDefinition. Ensure it is registered."
            )
            # return QueryDefinition(**v)
        return v

    def __str__(self):
        return f"Query(task_id={self.task_id}, query_str={self.query_str}, dependencies={self.dependencies}, node_type={self.node_type}, definition={self.definition})"

    def __repr__(self):
        return self.__str__()


class QueryPlan(BaseModel):
    """
    Container class representing a tree of questions to ask a question answer system.
    and its dependencies. Make sure every question is in the tree, and every question is asked only once.
    """

    nodes: list[Query] = Field(..., description="The original question we are asking")

    def dependencies(self, idz: list[str]) -> list[Query]:
        """
        Returns the dependencies of the query with the given id.
        """
        return [q for q in self.nodes if q.task_id in idz]


@dataclass
class PlannerInfo:
    """
    A transfer object that holds relevant information for constructing
    a structured query execution graph for document annotation.

    Example of usage:
        planner_info = PlannerInfo(
            name="complex",
            steps=["SEGMENT", "FIELD_EXTRACTION", "TABLE_EXTRACTION"]
        )
        # This object can then be handed off to a query_planner function.
    """

    name: str
    base_id: str  # UUID7 identifier
    current_id: int = 0  # Node counter
    steps: List[str] = field(default_factory=list)
    metadata: Any = None
