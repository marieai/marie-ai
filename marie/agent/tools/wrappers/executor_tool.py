"""ExecutorTool - Wrap Marie Executors as agent tools.

This module provides tools that wrap Marie Executors, enabling agents
to call executor endpoints directly without HTTP overhead.
"""

from __future__ import annotations

import asyncio
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.executor.marie_executor import MarieExecutor

logger = MarieLogger("marie.agent.tools.wrappers.executor")


class ExecutorToolInput(BaseModel):
    """Default input schema for executor tools."""

    input: str = Field(..., description="Input data for the executor")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the executor",
    )


class ExecutorTool(AgentTool):
    """Tool that wraps a Marie Executor endpoint.

    Enables agents to call executor endpoints directly using Python calls
    rather than going through HTTP/gRPC, providing zero-overhead integration.

    Example:
        ```python
        from marie.executor.text import TextExtractionExecutor

        # Create executor
        executor = TextExtractionExecutor()

        # Wrap as tool
        tool = ExecutorTool.from_executor(
            executor=executor,
            endpoint="/document/extract",
            name="extract_text",
            description="Extract text from documents",
        )

        # Use in agent
        result = await tool.acall(input="/path/to/document.pdf")
        ```
    """

    def __init__(
        self,
        executor: "MarieExecutor",
        endpoint: str,
        metadata: ToolMetadata,
        input_transformer: Optional[Callable[[Any], tuple]] = None,
        output_transformer: Optional[Callable[[Any], str]] = None,
    ):
        """Initialize ExecutorTool.

        Args:
            executor: The Marie executor instance
            endpoint: Endpoint path (e.g., "/document/extract")
            metadata: Tool metadata
            input_transformer: Optional function to transform input
            output_transformer: Optional function to transform output
        """
        self._executor = executor
        self._endpoint = endpoint
        self._metadata = metadata
        self._input_transformer = input_transformer or self._default_input_transformer
        self._output_transformer = (
            output_transformer or self._default_output_transformer
        )
        self._endpoint_method = self._resolve_endpoint()

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def _resolve_endpoint(self) -> Callable:
        """Resolve the endpoint method from the executor.

        Returns:
            The endpoint method callable

        Raises:
            ValueError: If endpoint not found
        """
        # Look for method with matching @requests decorator
        for attr_name in dir(self._executor):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self._executor, attr_name)
            if not callable(attr):
                continue

            # Check for requests decorator info
            if hasattr(attr, "__requests__"):
                if attr.__requests__.get("on") == self._endpoint:
                    return attr

            # Alternative: check __wrapped__ for decorated methods
            if hasattr(attr, "__wrapped__"):
                wrapped = attr.__wrapped__
                if hasattr(wrapped, "__requests__"):
                    if wrapped.__requests__.get("on") == self._endpoint:
                        return attr

        # Fallback: try to find by naming convention
        endpoint_name = self._endpoint.strip("/").replace("/", "_").replace("-", "_")
        if hasattr(self._executor, endpoint_name):
            return getattr(self._executor, endpoint_name)

        raise ValueError(
            f"Could not find endpoint '{self._endpoint}' on executor "
            f"{type(self._executor).__name__}"
        )

    def _default_input_transformer(
        self,
        input_data: Any,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Default input transformer.

        Args:
            input_data: Input data (string, dict, or structured)
            parameters: Additional parameters

        Returns:
            Tuple of (docs, parameters) for executor
        """
        from docarray import DocList

        # Handle different input types
        from docarray.documents import TextDoc

        from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler

        if isinstance(input_data, str):
            # Assume it's a file path or text content
            docs = DocList[TextDoc]([TextDoc(text=input_data)])
        elif isinstance(input_data, dict):
            text = input_data.get("text", input_data.get("input", str(input_data)))
            docs = DocList[TextDoc]([TextDoc(text=text)])
        elif hasattr(input_data, "__iter__"):
            docs = input_data
        else:
            docs = DocList[TextDoc]([TextDoc(text=str(input_data))])

        return docs, parameters or {}

    def _default_output_transformer(self, result: Any) -> str:
        """Default output transformer.

        Args:
            result: Executor result

        Returns:
            String representation
        """
        if result is None:
            return ""

        if isinstance(result, str):
            return result

        # Handle DocList/DocArray results
        if hasattr(result, "__iter__") and hasattr(result, "__len__"):
            outputs = []
            for doc in result:
                if hasattr(doc, "text") and doc.text:
                    outputs.append(doc.text)
                elif hasattr(doc, "content"):
                    outputs.append(str(doc.content))
                else:
                    outputs.append(str(doc))
            return "\n".join(outputs) if outputs else str(result)

        # Try JSON serialization
        try:
            if hasattr(result, "model_dump"):
                return json.dumps(result.model_dump(), ensure_ascii=False)
            elif hasattr(result, "dict"):
                return json.dumps(result.dict(), ensure_ascii=False)
            else:
                return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    def call(self, **kwargs: Any) -> ToolOutput:
        """Execute the executor endpoint synchronously.

        Args:
            **kwargs: Input arguments

        Returns:
            ToolOutput with result
        """
        from marie.helper import run_async

        return run_async(self.acall(**kwargs))

    async def acall(self, **kwargs: Any) -> ToolOutput:
        """Execute the executor endpoint asynchronously.

        Args:
            **kwargs: Input arguments including:
                - input: Main input data
                - parameters: Additional executor parameters
                - Any other kwargs are passed to the executor

        Returns:
            ToolOutput with result
        """
        input_data = kwargs.pop("input", kwargs.pop("query", ""))
        parameters = kwargs.pop("parameters", {})
        parameters.update(kwargs)

        try:
            # Transform input
            docs, params = self._input_transformer(input_data, parameters)

            # Call executor endpoint
            if asyncio.iscoroutinefunction(self._endpoint_method):
                result = await self._endpoint_method(docs=docs, parameters=params)
            else:
                result = self._endpoint_method(docs=docs, parameters=params)

            # Transform output
            output_str = self._output_transformer(result)

            return ToolOutput(
                content=output_str,
                tool_name=self.name,
                raw_input={"input": input_data, "parameters": parameters},
                raw_output=result,
            )

        except Exception as e:
            logger.error(f"Executor tool '{self.name}' failed: {e}")
            return ToolOutput(
                content=f"Error: {str(e)}",
                tool_name=self.name,
                raw_input={"input": input_data, "parameters": parameters},
                raw_output=None,
                is_error=True,
            )

    @classmethod
    def from_executor(
        cls,
        executor: "MarieExecutor",
        endpoint: str = "/",
        name: Optional[str] = None,
        description: Optional[str] = None,
        fn_schema: Optional[Type[BaseModel]] = None,
        input_transformer: Optional[Callable] = None,
        output_transformer: Optional[Callable] = None,
    ) -> "ExecutorTool":
        """Create an ExecutorTool from an executor instance.

        Args:
            executor: The executor instance
            endpoint: Endpoint to wrap
            name: Tool name (defaults to executor class name + endpoint)
            description: Tool description
            fn_schema: Input schema
            input_transformer: Custom input transformer
            output_transformer: Custom output transformer

        Returns:
            Configured ExecutorTool
        """
        # Generate default name from executor and endpoint
        executor_name = type(executor).__name__
        if name is None:
            endpoint_suffix = endpoint.strip("/").replace("/", "_").replace("-", "_")
            name = (
                f"{executor_name}_{endpoint_suffix}"
                if endpoint_suffix
                else executor_name
            )

        # Generate description from executor docstring
        if description is None:
            description = (
                (executor.__doc__ or f"Execute {endpoint} on {executor_name}")
                .split("\n")[0]
                .strip()
            )

        metadata = ToolMetadata(
            name=name,
            description=description,
            fn_schema=fn_schema or ExecutorToolInput,
        )

        return cls(
            executor=executor,
            endpoint=endpoint,
            metadata=metadata,
            input_transformer=input_transformer,
            output_transformer=output_transformer,
        )


class DocumentExtractionTool(ExecutorTool):
    """Specialized tool for document extraction executors.

    Provides a convenient interface for text extraction tasks.
    """

    class InputSchema(BaseModel):
        """Input schema for document extraction."""

        file_path: str = Field(..., description="Path to the document file")
        ref_id: Optional[str] = Field(
            default=None,
            description="Reference ID for tracking",
        )
        options: Dict[str, Any] = Field(
            default_factory=dict,
            description="Extraction options",
        )

    @classmethod
    def from_executor(
        cls,
        executor: "MarieExecutor",
        endpoint: str = "/document/extract",
        name: str = "extract_document",
        description: str = "Extract text and structure from documents",
        **kwargs: Any,
    ) -> "DocumentExtractionTool":
        """Create a document extraction tool."""
        return super().from_executor(
            executor=executor,
            endpoint=endpoint,
            name=name,
            description=description,
            fn_schema=cls.InputSchema,
            **kwargs,
        )


class JobStatusTool(AgentTool):
    """Tool to check job status from the scheduler.

    Provides agents with the ability to monitor background job progress.
    """

    class InputSchema(BaseModel):
        """Input schema for job status check."""

        job_id: str = Field(..., description="Job ID to check status for")

    def __init__(
        self,
        job_manager: Any,  # JobManager type
        name: str = "check_job_status",
        description: str = "Check the status of a background job",
    ):
        """Initialize JobStatusTool.

        Args:
            job_manager: JobManager instance
            name: Tool name
            description: Tool description
        """
        self._job_manager = job_manager
        self._metadata = ToolMetadata(
            name=name,
            description=description,
            fn_schema=self.InputSchema,
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, job_id: str, **kwargs: Any) -> ToolOutput:
        """Check job status synchronously."""
        from marie.helper import run_async

        return run_async(self.acall(job_id=job_id, **kwargs))

    async def acall(self, job_id: str, **kwargs: Any) -> ToolOutput:
        """Check job status asynchronously.

        Args:
            job_id: Job ID to check

        Returns:
            ToolOutput with job status
        """
        try:
            status = await self._job_manager.get_job_info(job_id)

            if status is None:
                return ToolOutput(
                    content=f"Job '{job_id}' not found",
                    tool_name=self.name,
                    raw_input={"job_id": job_id},
                    raw_output=None,
                )

            # Format status
            if hasattr(status, "model_dump"):
                status_dict = status.model_dump()
            elif hasattr(status, "dict"):
                status_dict = status.dict()
            else:
                status_dict = {"status": str(status)}

            return ToolOutput(
                content=json.dumps(status_dict, ensure_ascii=False, indent=2),
                tool_name=self.name,
                raw_input={"job_id": job_id},
                raw_output=status_dict,
            )

        except Exception as e:
            return ToolOutput(
                content=f"Error checking job status: {e}",
                tool_name=self.name,
                raw_input={"job_id": job_id},
                raw_output=None,
                is_error=True,
            )
