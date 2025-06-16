import enum
import importlib
import warnings
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field, field_validator

from marie.logging_core.predefined import default_logger as logger
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
    """Registry for query planner functions with wheel support."""

    _plans: Dict[str, Callable] = {}
    _external_modules_loaded: bool = False

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

        logger.info(f"Registering query planner function : {name}")

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
    def initialize_from_config(cls, query_planners_conf) -> Dict[str, Any]:
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
            'total_planners': len(cls._plans),
            'wheel_results': {},
        }

        # Handle wheel directories
        wheel_directories = getattr(query_planners_conf, 'wheel_directories', [])
        wheel_watch = getattr(query_planners_conf, 'watch_wheels', True)

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
