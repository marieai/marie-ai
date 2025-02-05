import enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

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


class QueryDefinitionXXXXXXX(BaseModel):
    """
    Represents a task that executes via a registered execution method.
    """

    method: str = Field(
        ..., description="Type of execution method (LLM, Python, API, etc.)."
    )
    params: Dict[str, Optional[str]] = Field(
        default={}, description="Input parameters required for execution."
    )


class QueryDefinition(BaseModel):
    """
    Abstract base class for query definitions. Represents a task executed via a specific method.
    Type of execution method (LLM, Python, API, etc.). Input parameters required for execution.
    """

    method: str
    params: dict

    def validate_params(self):
        """Validate the parameters for the query."""
        raise NotImplementedError("Subclasses must implement validate_params.")


class NoopQueryDefinition(QueryDefinition):
    """
    Represents a NOOP (no operation) query definition.
    It can be used to group tasks in a DAG. The task is evaluated by the scheduler but never processed by the executor.
    """

    method: str = "NOOP"
    params: dict = Field(default_factory=lambda: {"layout": None})

    def validate_params(self):
        if "layout" not in self.params or self.params["layout"] is None:
            raise ValueError("NOOP queries must have a 'layout' parameter.")


class LlmQueryDefinition(QueryDefinition):
    """
    Represents an LLM (language model) query definition.
    """

    method: str = "LLM"
    params: dict = Field(default_factory=lambda: {"layout": None, "roi": None})

    def validate_params(self):
        if "layout" not in self.params or self.params["layout"] is None:
            raise ValueError("LLM queries must have a 'layout' parameter.")
        if "roi" not in self.params or self.params["roi"] is None:
            raise ValueError(
                "LLM queries must specify a 'roi' parameter (e.g., 'start', 'end')."
            )


class PythonFunctionQueryDefinition(QueryDefinition):
    """
    Represents a Python function execution query definition.
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

    definition: Optional[QueryDefinition] = Field(
        default=None,
        description="Definition of the query to be executed.",
    )

    def __str__(self):
        return f"Query {self.task_id}: {self.question}"


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
