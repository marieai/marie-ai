"""
Asset execution context for Marie-AI.

Provides runtime context to asset functions during execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from marie.logging_core.logger import MarieLogger


@dataclass
class RunInfo:
    """Information about the current job execution run."""

    run_id: str  # Unique job execution identifier
    tags: Dict[str, str] = field(default_factory=dict)
    run_config: Dict[str, Any] = field(default_factory=dict)


class AssetExecutionContext:
    """
    Runtime context passed to asset functions during execution.

    Provides runtime information about the current job execution, DAG position,
    and asset tracking details. The context is the first parameter passed to
    all @asset decorated functions.

    Key properties are accessed via `context.run.run_id`, `context.partition_key`,
    `context.log`, etc.

    Example:
        ```python
        @asset(key="ocr/text")
        def extract_text(context: AssetExecutionContext, docs):
            context.log.info(f"Processing job {context.run.run_id}")

            if context.has_partition_key:
                context.log.info(f"Partition: {context.partition_key}")

            if context.is_dag_execution:
                context.log.info(f"DAG: {context.dag_id}, Node: {context.node_task_id}")

            return process_docs(docs)
        ```
    """

    def __init__(
        self,
        job_id: str,
        dag_id: Optional[str] = None,
        node_task_id: Optional[str] = None,
        partition_key: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None,
        asset_key: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize AssetExecutionContext.

        Args:
            job_id: Unique job execution identifier
            dag_id: DAG identifier (if part of a DAG)
            node_task_id: Task ID within the DAG
            partition_key: Partition key for partitioned assets
            run_config: Runtime configuration dictionary
            resources: Available resources (storage, clients, etc.)
            asset_key: Current asset key being executed
            tags: Run tags
        """
        # Build RunInfo with execution details
        self._run = RunInfo(run_id=job_id, tags=tags or {}, run_config=run_config or {})

        self._dag_id = dag_id
        self._node_task_id = node_task_id
        self._partition_key = partition_key
        self._resources = resources or {}
        self._asset_key = asset_key
        self._log = MarieLogger("AssetContext")

    # ========================================================================
    # Core API - Run Information and Logging
    # ========================================================================

    @property
    def run(self) -> RunInfo:
        """
        The RunInfo object corresponding to the current execution.

        Provides access to run_id (job execution ID), configuration, and tags.

        Example:
            ```python
            @asset
            def my_asset(context):
                context.log.info(f"Run ID: {context.run.run_id}")
                context.log.info(f"Tags: {context.run.tags}")
            ```
        """
        return self._run

    @property
    def log(self) -> MarieLogger:
        """
        The log manager available in the execution context.

        Logs are viewable in Marie-AI execution logs and can be filtered by job ID.

        Example:
            ```python
            @asset
            def logger_example(context):
                context.log.info("Info level message")
                context.log.warning("Warning message")
                context.log.error("Error message")
            ```
        """
        return self._log

    @property
    def asset_key(self) -> Optional[str]:
        """
        The asset key for the currently executing asset.

        Example:
            ```python
            @asset(key="ocr/text")
            def extract_text(context):
                context.log.info(f"Materializing {context.asset_key}")
            ```
        """
        return self._asset_key

    @property
    def has_partition_key(self) -> bool:
        """
        Whether the current run is a partitioned run.

        Example:
            ```python
            @asset
            def partitioned_asset(context):
                if context.has_partition_key:
                    context.log.info(f"Processing partition: {context.partition_key}")
            ```
        """
        return self._partition_key is not None

    @property
    def partition_key(self) -> str:
        """
        The partition key for the currently executing partition.

        Raises:
            ValueError: If no partition key is set

        Example:
            ```python
            @asset(partitions_def=daily_partitions)
            def daily_asset(context):
                date = context.partition_key  # e.g., "2025-11-11"
                return process_for_date(date)
            ```
        """
        if self._partition_key is None:
            raise ValueError("No partition key set for this execution")
        return self._partition_key

    # ========================================================================
    # Marie-AI specific properties (DAG execution)
    # ========================================================================

    @property
    def dag_id(self) -> Optional[str]:
        """
        The DAG identifier if this execution is part of a DAG.

        Marie-AI specific: Used for DAG-based execution tracking.
        """
        return self._dag_id

    @property
    def node_task_id(self) -> Optional[str]:
        """
        The node task ID within the DAG.

        Marie-AI specific: Identifies the node within a DAG execution.
        """
        return self._node_task_id

    @property
    def is_dag_execution(self) -> bool:
        """Check if executing as part of a DAG."""
        return self._dag_id is not None

    # ========================================================================
    # Resource access
    # ========================================================================

    def get_resource(self, name: str) -> Any:
        """
        Get a resource by name.

        Args:
            name: Resource name

        Returns:
            Resource object

        Raises:
            KeyError: If resource not found

        Example:
            ```python
            @asset
            def my_asset(context):
                s3 = context.get_resource("s3")
                db = context.get_resource("database")
            ```
        """
        if name not in self._resources:
            raise KeyError(f"Resource '{name}' not available in context")
        return self._resources[name]

    def has_resource(self, name: str) -> bool:
        """Check if a resource is available."""
        return name in self._resources

    # ========================================================================
    # Factory methods
    # ========================================================================

    @classmethod
    def from_parameters(
        cls,
        parameters: Dict[str, Any],
        resources: Optional[Dict[str, Any]] = None,
        asset_key: Optional[str] = None,
    ) -> "AssetExecutionContext":
        """
        Create context from executor parameters.

        Args:
            parameters: Parameters dict from executor request
            resources: Optional resources dict
            asset_key: Optional asset key being executed

        Returns:
            AssetExecutionContext instance

        Example:
            ```python
            def my_executor_method(self, docs, parameters):
                context = AssetExecutionContext.from_parameters(parameters)
                result = my_asset_function(context, docs)
                return result
            ```
        """
        return cls(
            job_id=parameters.get("job_id", "unknown"),
            dag_id=parameters.get("dag_id"),
            node_task_id=parameters.get("node_task_id"),
            partition_key=parameters.get("partition_key"),
            run_config=parameters.get("run_config", {}),
            resources=resources or {},
            asset_key=asset_key,
            tags=parameters.get("tags", {}),
        )

    # ========================================================================
    # Utility methods
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "run_id": self.run.run_id,
            "dag_id": self.dag_id,
            "node_task_id": self.node_task_id,
            "partition_key": self._partition_key,
            "asset_key": self.asset_key,
            "run_config": self.run.run_config,
            "tags": self.run.tags,
        }


class AssetMaterializationContext(AssetExecutionContext):
    """
    Extended context for asset materialization with storage handler access.

    Provides additional functionality for asset tracking and persistence.
    """

    def __init__(
        self,
        job_id: str,
        storage_handler=None,
        asset_tracker=None,
        dag_id: Optional[str] = None,
        node_task_id: Optional[str] = None,
        partition_key: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None,
        asset_key: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize materialization context."""
        super().__init__(
            job_id=job_id,
            dag_id=dag_id,
            node_task_id=node_task_id,
            partition_key=partition_key,
            run_config=run_config,
            resources=resources,
            asset_key=asset_key,
            tags=tags,
        )
        self.storage_handler = storage_handler
        self.asset_tracker = asset_tracker

    @classmethod
    def from_executor(
        cls,
        parameters: Dict[str, Any],
        storage_handler=None,
        asset_tracker=None,
        resources: Optional[Dict[str, Any]] = None,
        asset_key: Optional[str] = None,
    ) -> "AssetMaterializationContext":
        """
        Create materialization context from executor.

        Args:
            parameters: Parameters dict from executor request
            storage_handler: Storage handler instance
            asset_tracker: Asset tracker instance
            resources: Optional resources dict
            asset_key: Optional asset key being executed

        Returns:
            AssetMaterializationContext instance
        """
        return cls(
            job_id=parameters.get("job_id", "unknown"),
            dag_id=parameters.get("dag_id"),
            node_task_id=parameters.get("node_task_id"),
            partition_key=parameters.get("partition_key"),
            run_config=parameters.get("run_config", {}),
            resources=resources or {},
            storage_handler=storage_handler,
            asset_tracker=asset_tracker,
            asset_key=asset_key,
            tags=parameters.get("tags", {}),
        )

    def materialize_asset(
        self, asset_key: str, data: Any, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record asset materialization.

        Args:
            asset_key: Asset key to materialize
            data: Asset data/result
            metadata: Additional metadata for this materialization
        """
        if not self.asset_tracker:
            self.log.warning("Asset tracker not available, skipping materialization")
            return

        # This would integrate with the asset tracker
        self.log.debug(f"Materializing asset: {asset_key}")
        # Implementation would call asset_tracker.record_materializations()


def build_asset_context(
    parameters: Dict[str, Any],
    storage_handler=None,
    asset_tracker=None,
    resources: Optional[Dict[str, Any]] = None,
    asset_key: Optional[str] = None,
) -> AssetExecutionContext:
    """
    Build appropriate asset context from parameters.

    Factory function for creating context objects from executor parameters.

    Args:
        parameters: Parameters dict from executor (containing job_id, dag_id, etc.)
        storage_handler: Optional storage handler for asset persistence
        asset_tracker: Optional asset tracker for materialization tracking
        resources: Optional resources dict (databases, S3 clients, etc.)
        asset_key: Optional asset key being executed

    Returns:
        AssetExecutionContext or AssetMaterializationContext

    Example:
        ```python
        # In executor
        @requests(on="/document/process")
        def process(self, docs, parameters):
            # Build context with storage/tracking
            context = build_asset_context(
                parameters=parameters,
                storage_handler=self.storage_handler if self.storage_enabled else None,
                asset_tracker=self.asset_tracker if self.asset_tracking_enabled else None,
                asset_key="ocr/text",
            )

            # Call asset function
            result = my_asset_function(context, docs)

            return result
        ```
    """
    if storage_handler or asset_tracker:
        return AssetMaterializationContext.from_executor(
            parameters=parameters,
            storage_handler=storage_handler,
            asset_tracker=asset_tracker,
            resources=resources,
            asset_key=asset_key,
        )
    return AssetExecutionContext.from_parameters(
        parameters, resources=resources, asset_key=asset_key
    )
