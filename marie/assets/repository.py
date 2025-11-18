"""Asset Repository - query layer for asset data."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from marie.logging_core.logger import MarieLogger

from .models import (
    AssetInfo,
    AssetLatestInfo,
    AssetLineage,
    MaterializationInfo,
    NodeMaterializationStatus,
    UpstreamAssetInfo,
)


class AssetRepository:
    """
    Repository for querying asset data.

    Provides read-only access to:
    - Asset registry
    - Materialization history
    - Lineage information
    """

    def __init__(self, config: Dict[str, Any], max_workers: int = 2):
        """
        Initialize asset repository.

        Args:
            config: Database configuration
            max_workers: Number of thread pool workers
        """
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.config = config
        self._db_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="asset-repo"
        )
        self._loop = asyncio.get_event_loop()

        # Import PostgresqlMixin for connection management
        from marie.storage.database.postgres import PostgresqlMixin

        self._pg_mixin = PostgresqlMixin()
        self._pg_mixin._setup_storage(config, connection_only=True)

    def _get_connection(self):
        """Get database connection."""
        return self._pg_mixin._get_connection()

    def _close_connection(self, conn):
        """Close database connection."""
        return self._pg_mixin._close_connection(conn)

    async def get_asset_info(self, asset_key: str) -> Optional[AssetInfo]:
        """
        Get asset registry information.

        Args:
            asset_key: Asset key

        Returns:
            AssetInfo if found, None otherwise
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id, asset_key, namespace, kind, description, tags, created_at, updated_at
                    FROM marie_scheduler.asset_registry
                    WHERE asset_key = %s
                    """,
                    (asset_key,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return AssetInfo(
                    id=row[0],
                    asset_key=row[1],
                    namespace=row[2],
                    kind=row[3],
                    description=row[4],
                    tags=row[5] or {},
                    created_at=row[6],
                    updated_at=row[7],
                )

            finally:
                if cursor:
                    cursor.close()
                if conn:
                    self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def get_latest_version(self, asset_key: str) -> Optional[AssetLatestInfo]:
        """
        Get latest version of an asset.

        Args:
            asset_key: Asset key

        Returns:
            AssetLatestInfo or None
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT asset_key, latest_version, latest_at, partition_key
                    FROM marie_scheduler.asset_latest
                    WHERE asset_key = %s
                    """,
                    (asset_key,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return AssetLatestInfo(
                    asset_key=row[0],
                    version=row[1],
                    latest_at=row[2],
                    partition_key=row[3],
                )

            finally:
                if cursor:
                    cursor.close()
                if conn:
                    self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def get_materialization_history(
        self, asset_key: str, limit: int = 10
    ) -> List[MaterializationInfo]:
        """
        Get materialization history for an asset.

        Args:
            asset_key: Asset key
            limit: Maximum number of records

        Returns:
            List of MaterializationInfo
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT * FROM marie_scheduler.get_asset_history(%s, %s)
                    """,
                    (asset_key, limit),
                )

                results = []
                for row in cursor.fetchall():
                    results.append(
                        MaterializationInfo(
                            id=row[0],
                            storage_event_id=None,
                            asset_key=asset_key,
                            asset_version=row[1],
                            job_id=row[2],
                            dag_id=row[3],
                            node_task_id=row[4],
                            partition_key=row[5],
                            size_bytes=row[6],
                            checksum=row[7],
                            uri=row[8],
                            metadata={},
                            created_at=row[9],
                        )
                    )

                return results

            finally:
                if cursor:
                    cursor.close()
                if conn:
                    self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def get_lineage(self, asset_key: str) -> AssetLineage:
        """
        Get lineage for an asset.

        Args:
            asset_key: Asset key

        Returns:
            AssetLineage with upstream dependencies
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Get direct upstream
                cursor.execute(
                    """
                    SELECT DISTINCT
                        al.upstream_asset_key,
                        al.upstream_version,
                        al.upstream_partition_key
                    FROM marie_scheduler.asset_materialization am
                    JOIN marie_scheduler.asset_lineage al ON al.materialization_id = am.id
                    WHERE am.asset_key = %s
                    ORDER BY am.created_at DESC
                    LIMIT 100
                    """,
                    (asset_key,),
                )

                upstream = []
                for row in cursor.fetchall():
                    upstream.append(
                        UpstreamAssetInfo(
                            asset_key=row[0],
                            version=row[1],
                            partition_key=row[2],
                        )
                    )

                return AssetLineage(asset_key=asset_key, upstream=upstream)

            finally:
                if cursor:
                    cursor.close()
                if conn:
                    self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def get_node_status(
        self, dag_id: str, node_task_id: str
    ) -> Optional[NodeMaterializationStatus]:
        """
        Get materialization status for a DAG node.

        Args:
            dag_id: DAG ID
            node_task_id: Node task ID

        Returns:
            NodeMaterializationStatus or None
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT
                        dag_id,
                        dag_name,
                        node_task_id,
                        expected_assets,
                        materialized_assets,
                        required_assets,
                        materialized_required,
                        all_required_materialized,
                        missing_required_assets
                    FROM marie_scheduler.node_materialization_status
                    WHERE dag_id = %s AND node_task_id = %s
                    """,
                    (dag_id, node_task_id),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return NodeMaterializationStatus(
                    dag_id=row[0],
                    dag_name=row[1],
                    node_task_id=row[2],
                    expected_assets=row[3],
                    materialized_assets=row[4],
                    required_assets=row[5],
                    materialized_required=row[6],
                    all_required_materialized=row[7],
                    missing_required_assets=row[8] or [],
                )

            finally:
                if cursor:
                    cursor.close()
                if conn:
                    self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    async def get_dag_status(self, dag_id: str) -> List[NodeMaterializationStatus]:
        """
        Get materialization status for all nodes in a DAG.

        Args:
            dag_id: DAG ID

        Returns:
            List of NodeMaterializationStatus for each node
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT
                        dag_id,
                        dag_name,
                        node_task_id,
                        expected_assets,
                        materialized_assets,
                        required_assets,
                        materialized_required,
                        all_required_materialized,
                        missing_required_assets
                    FROM marie_scheduler.node_materialization_status
                    WHERE dag_id = %s
                    ORDER BY node_task_id
                    """,
                    (dag_id,),
                )

                results = []
                for row in cursor.fetchall():
                    results.append(
                        NodeMaterializationStatus(
                            dag_id=row[0],
                            dag_name=row[1],
                            node_task_id=row[2],
                            expected_assets=row[3],
                            materialized_assets=row[4],
                            required_assets=row[5],
                            materialized_required=row[6],
                            all_required_materialized=row[7],
                            missing_required_assets=row[8] or [],
                        )
                    )

                return results

            finally:
                if cursor:
                    cursor.close()
                if conn:
                    self._close_connection(conn)

        return await self._loop.run_in_executor(self._db_executor, db_call)

    def cleanup(self):
        """Cleanup resources."""
        self._db_executor.shutdown(wait=True)
