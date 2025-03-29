import enum
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field, field_validator

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


class QueryPlanRegistry:
    """Registry for query planner functions."""

    _plans: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str = None):
        """
        Decorator to register a query planner function.

        Usage:
            @QueryPlannerRegistry.register("my_planner")
            def my_query_planner(planner_info):
                # planner implementation
                return plan

        :param name: The name to register the planner under. If None, uses the function name.
        :return: Decorator function
        """

        def decorator(func: Callable) -> Callable:
            planner_name = name or func.__name__

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            if planner_name in cls._plans:
                raise ValueError(
                    f"Query planner '{planner_name}' is already registered"
                )

            cls._plans[planner_name] = wrapper
            return wrapper

        return decorator

    @classmethod
    def get(cls, planner_name: str) -> Callable:
        """
        Get a query planner by name.

        :param planner_name: The name of the query planner to get.
        :return: The query planner function.
        :raises ValueError: If the query planner is not registered.
        """
        if planner_name in cls._plans:
            return cls._plans[planner_name]

        raise ValueError(f"Unknown query planner: {planner_name}")

    @classmethod
    def list_planners(cls) -> list[str]:
        """Return a list of all registered planner names."""
        return list(cls._plans.keys())


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
    params: dict = Field(default_factory=lambda: {"layout": None, "roi": None})

    def validate_params(self):
        if "layout" not in self.params or self.params["layout"] is None:
            raise ValueError("LLM queries must have a 'layout' parameter.")
        if "roi" not in self.params or self.params["roi"] is None:
            raise ValueError(
                "LLM queries must specify a 'roi' parameter (e.g., 'start', 'end')."
            )


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

    # def __str__(self):
    #     return f"Query {self.task_id}"
    #
    # def __repr__(self):
    #     return f"Query {self.task_id}"


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
