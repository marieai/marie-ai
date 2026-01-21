"""
ExecutionContext - Simplified data access for annotators based on upstream results.

This module provides access to upstream task results via RunContext.
It does NOT evaluate conditions automatically - that's left to custom evaluators
or the annotators themselves.

Example YAML configuration (simplified):
    annotators:
      remarks:
        annotator_type: "llm"
        model_config:
          model_name: remark_codes
          prompt_path: "./remarks.j2"

        # Just list dependencies - no condition evaluation
        execution_context:
          depends_on:
            - tables
            - claims

Or with backward-compatible dict format:
        execution_context:
          depends_on:
            - task_id: tables
            - task_id: claims
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie_kernel.context import RunContext

    from marie.extract.structures.unstructured_document import UnstructuredDocument


@dataclass
class ExecutionContextConfig:
    """
    Simple config - just list upstream task dependencies.

    Attributes:
        depends_on: List of upstream task IDs (annotator names)
    """

    depends_on: List[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ExecutionContextConfig":
        """
        Create ExecutionContextConfig from a configuration dictionary.

        Supports both simple list format and dict format for backward compatibility.

        Args:
            config: Dictionary with execution context configuration

        Returns:
            ExecutionContextConfig instance

        Examples:
            # Simple list format (preferred)
            config = {"depends_on": ["tables", "claims"]}

            # Dict format (backward compatible)
            config = {
                "depends_on": [
                    {"task_id": "tables"},
                    {"task_id": "claims"}
                ]
            }
        """
        depends_on_raw = config.get("depends_on", [])
        depends_on: List[str] = []

        for item in depends_on_raw:
            if isinstance(item, str):
                # Simple string format
                depends_on.append(item)
            elif isinstance(item, dict):
                # Dict format for backward compatibility
                task_id = item.get("task_id", "")
                if task_id:
                    depends_on.append(task_id)

        return cls(depends_on=depends_on)


class ExecutionContext:
    """
    Provides access to upstream task results. No automatic condition evaluation.

    This class loads data from upstream tasks and makes it available for
    the annotator or custom evaluators to use as needed.

    Example:
        ```python
        config = ExecutionContextConfig.from_config({"depends_on": ["tables"]})
        exec_ctx = ExecutionContext(config, run_context)

        # Access upstream data
        tables_data = exec_ctx.get_upstream_data("tables")
        if tables_data:
            # Custom logic based on actual output structure
            ...
        ```
    """

    # Standard key used for annotator results in RunContext
    ANNOTATOR_RESULTS_KEY = "ANNOTATOR_RESULTS"

    def __init__(
        self,
        config: ExecutionContextConfig,
        run_context: "RunContext",
    ):
        """
        Initialize ExecutionContext.

        Args:
            config: Execution context configuration with dependencies
            run_context: RunContext for accessing upstream task results
        """
        self.config = config
        self.ctx = run_context
        self.logger = MarieLogger(context=self.__class__.__name__)
        self._upstream_data: Dict[str, Any] = {}
        self._load_upstream_data()

    def _load_upstream_data(self) -> None:
        """Load data from all upstream tasks."""
        for task_id in self.config.depends_on:
            try:
                data = self.ctx.get(
                    self.ANNOTATOR_RESULTS_KEY,
                    from_task=task_id,
                    default=None,
                )
                if data is not None:
                    self._upstream_data[task_id] = data
                    self.logger.info(f"Loaded upstream data from '{task_id}'")
                else:
                    self.logger.warning(f"No data found for upstream task '{task_id}'")
            except Exception as e:
                self.logger.error(
                    f"Error loading data from upstream task '{task_id}': {e}",
                    exc_info=True,
                )

    def get_upstream_data(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get data from a specific upstream task.

        Args:
            task_id: The upstream task ID to get data from

        Returns:
            The upstream data if available, None otherwise
        """
        return self._upstream_data.get(task_id)

    def get_all_upstream_data(self) -> Dict[str, Any]:
        """
        Get all loaded upstream data.

        Returns:
            Dictionary mapping task_id to upstream data
        """
        return self._upstream_data.copy()

    def has_upstream_data(self, task_id: str) -> bool:
        """
        Check if upstream task has data available.

        Args:
            task_id: The upstream task ID to check

        Returns:
            True if data is available for the task, False otherwise
        """
        return task_id in self._upstream_data

    @property
    def dependency_count(self) -> int:
        """Return the number of configured dependencies."""
        return len(self.config.depends_on)

    @property
    def loaded_count(self) -> int:
        """Return the number of successfully loaded upstream tasks."""
        return len(self._upstream_data)


class ExecutionContextEvaluator(ABC):
    """
    Abstract base class for evaluating execution context conditions.

    Users can implement custom evaluators to determine which pages should
    be processed based on upstream task results.

    Example:
        ```python
        class MyCustomEvaluator(ExecutionContextEvaluator):
            def __init__(self, exec_ctx: ExecutionContext):
                self.ctx = exec_ctx

            def get_eligible_pages(self, document: UnstructuredDocument) -> Set[int]:
                tables_data = self.ctx.get_upstream_data("tables")
                if not tables_data:
                    return set(range(document.page_count))

                # Custom logic based on actual output structure
                eligible = set()
                pages = tables_data.get("pages", {})
                for page_num, page_data in pages.items():
                    if self._page_has_relevant_data(page_data):
                        eligible.add(int(page_num) - 1)  # Convert to 0-indexed
                return eligible

            def should_skip(self, document: UnstructuredDocument) -> bool:
                return len(self.get_eligible_pages(document)) == 0
        ```
    """

    @abstractmethod
    def get_eligible_pages(self, document: "UnstructuredDocument") -> Set[int]:
        """
        Evaluate conditions and return set of eligible page numbers.

        Args:
            document: The document being processed

        Returns:
            Set of page numbers (0-indexed) that are eligible for processing
        """
        pass

    @abstractmethod
    def should_skip(self, document: "UnstructuredDocument") -> bool:
        """
        Determine if the annotator should be skipped entirely.

        Args:
            document: The document being processed

        Returns:
            True if annotator should be skipped, False otherwise
        """
        pass
