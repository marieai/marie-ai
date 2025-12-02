"""DAG Asset Mapper - Query upstream assets from actual lineage."""

from typing import Dict, List

from marie.logging_core.logger import MarieLogger


class DAGAssetMapper:
    """
    Simplified asset mapper that queries upstream assets from actual materialization lineage.

    No longer pre-registers assets - assets are tracked dynamically as they are materialized.
    This class now only provides utility methods for querying upstream asset relationships.
    """

    def __init__(self):
        """Initialize DAG Asset Mapper."""
        self.logger = MarieLogger(self.__class__.__name__).logger

    @staticmethod
    def get_upstream_assets_for_node(
        dag_id: str, node_task_id: str, get_connection_fn, close_connection_fn
    ) -> List[Dict[str, str]]:
        """
        Get upstream assets for a DAG node based on actual lineage.

        Queries the asset_lineage table to find what assets were actually consumed
        by this node, not what was pre-defined.

        Args:
            dag_id: DAG UUID
            node_task_id: Node task ID within the DAG
            get_connection_fn: Function to get DB connection
            close_connection_fn: Function to close DB connection

        Returns:
            List of dicts with keys: asset_key, latest_version, partition_key

        Example:
            >>> upstream = DAGAssetMapper.get_upstream_assets_for_node(
            ...     dag_id="abc-123",
            ...     node_task_id="classify",
            ...     get_connection_fn=storage._get_connection,
            ...     close_connection_fn=storage._close_connection
            ... )
            >>> # Returns: [{"asset_key": "ocr/text", "latest_version": "v:sha256:...", "partition_key": None}]
        """
        conn = None
        cursor = None
        try:
            conn = get_connection_fn()
            cursor = conn.cursor()

            # Call the SQL function which queries actual lineage
            cursor.execute(
                "SELECT * FROM marie_scheduler.get_upstream_assets_for_node(%s, %s)",
                (dag_id, node_task_id),
            )

            result = [
                {
                    "asset_key": row[0],
                    "latest_version": row[1],
                    "partition_key": row[2],
                }
                for row in cursor.fetchall()
            ]

            # Commit the transaction (even for SELECT queries)
            if conn:
                conn.commit()

            return result

        finally:
            if cursor:
                cursor.close()
            if conn:
                close_connection_fn(conn)
