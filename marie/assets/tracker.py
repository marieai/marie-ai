"""Asset tracker for recording multi-asset materializations."""

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import psycopg2.extras

from marie.logging_core.logger import MarieLogger

from .models import LineageInfo


class AssetTracker:
    """
    Tracks asset materializations using existing PostgreSQL storage.

    This class is responsible for:
    1. Recording asset materializations
    2. Tracking lineage (upstream dependencies)
    3. Updating latest pointers

    NOTE: This is for TRACKING only, not execution control.
    The scheduler controls job execution via job dependencies.
    """

    def __init__(self, storage_handler, storage_conf: Dict):
        """
        Initialize asset tracker.

        Args:
            storage_handler: PostgreSQL storage handler instance
            storage_conf: Storage configuration dict
        """
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.storage_handler = storage_handler
        self.storage_conf = storage_conf
        self._db_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="asset-tracker"
        )
        self._loop = asyncio.get_event_loop()

    def record_materializations(
        self,
        storage_event_id: Optional[int],
        assets: List[Dict[str, Any]],
        job_id: str,
        dag_id: Optional[str] = None,
        node_task_id: Optional[str] = None,
        partition_key: Optional[str] = None,
        upstream_assets: Optional[List[Tuple[str, str, str]]] = None,
    ):
        """
        Record multiple asset materializations from a single storage event.

        Args:
            storage_event_id: ID from storage table (optional)
            assets: List of asset dicts, e.g.:
                [
                    {
                        "asset_key": "ocr/text",
                        "version": "v:sha256:abc...",
                        "size_bytes": 1024,
                        "checksum": "sha256:...",
                        "kind": "text",
                        "metadata": {"language": "en"}
                    }
                ]
            job_id: Job ID
            dag_id: DAG ID (optional)
            node_task_id: Node task ID (optional)
            partition_key: Partition key (optional)
            upstream_assets: Upstream dependencies (same for all assets from this node)
                            List of (asset_key, version, partition_key) tuples
        """

        def db_call():
            conn = None
            cursor = None
            try:
                conn = self.storage_handler._get_connection()
                cursor = conn.cursor()

                materialization_ids = []

                for asset in assets:
                    asset_key = asset["asset_key"]

                    # 1) Ensure asset is registered
                    cursor.execute(
                        """
                        INSERT INTO marie_scheduler.asset_registry (asset_key, kind, tags)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (asset_key) DO UPDATE SET
                            updated_at = now()
                        RETURNING id
                        """,
                        (
                            asset_key,
                            asset.get("kind", "unknown"),
                            psycopg2.extras.Json(asset.get("metadata", {})),
                        ),
                    )

                    # 2) Record materialization
                    cursor.execute(
                        """
                        INSERT INTO marie_scheduler.asset_materialization
                          (storage_event_id, asset_key, asset_version, job_id, dag_id,
                           node_task_id, partition_key, size_bytes, checksum, uri, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (storage_event_id, asset_key)
                        WHERE storage_event_id IS NOT NULL
                        DO UPDATE SET
                            asset_version = EXCLUDED.asset_version,
                            size_bytes = EXCLUDED.size_bytes,
                            checksum = EXCLUDED.checksum,
                            metadata = EXCLUDED.metadata
                        RETURNING id
                        """,
                        (
                            storage_event_id,
                            asset_key,
                            asset.get("version"),
                            job_id,
                            dag_id,
                            node_task_id,
                            partition_key,
                            asset.get("size_bytes"),
                            asset.get("checksum"),
                            asset.get("uri"),
                            psycopg2.extras.Json(asset.get("metadata", {})),
                        ),
                    )
                    mat_id = cursor.fetchone()[0]
                    materialization_ids.append((mat_id, asset_key))

                    # 3) Update latest pointer
                    cursor.execute(
                        """
                        INSERT INTO marie_scheduler.asset_latest
                          (asset_key, latest_materialization_id, latest_version, latest_at, partition_key)
                        VALUES (%s, %s, %s, now(), %s)
                        ON CONFLICT (asset_key) DO UPDATE SET
                            latest_materialization_id = EXCLUDED.latest_materialization_id,
                            latest_version = EXCLUDED.latest_version,
                            latest_at = EXCLUDED.latest_at,
                            partition_key = EXCLUDED.partition_key
                        WHERE asset_latest.latest_at < EXCLUDED.latest_at
                        """,
                        (asset_key, mat_id, asset.get("version"), partition_key),
                    )

                # 4) Record lineage (same upstream for all assets)
                if upstream_assets:
                    for mat_id, asset_key in materialization_ids:
                        for (
                            upstream_key,
                            upstream_version,
                            upstream_partition,
                        ) in upstream_assets:
                            cursor.execute(
                                """
                                INSERT INTO marie_scheduler.asset_lineage
                                  (materialization_id, upstream_asset_key, upstream_version, upstream_partition_key)
                                VALUES (%s, %s, %s, %s)
                                """,
                                (
                                    mat_id,
                                    upstream_key,
                                    upstream_version,
                                    upstream_partition,
                                ),
                            )

                conn.commit()
                self.logger.debug(
                    f"Recorded {len(materialization_ids)} asset materializations for job {job_id}"
                )
                return materialization_ids

            except Exception as e:
                if conn:
                    conn.rollback()
                self.logger.error(
                    f"Error recording materializations: {e}", exc_info=True
                )
                raise
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    self.storage_handler._close_connection(conn)

        # Run in thread pool
        return self._loop.run_in_executor(self._db_executor, db_call)

    @staticmethod
    def compute_version(*parts: bytes) -> str:
        """
        Compute content-addressed version from parts.

        Args:
            *parts: Byte strings to hash

        Returns:
            Version string like "v:sha256:abc123..."
        """
        h = hashlib.sha256()
        for part in parts:
            if part:
                h.update(part)
        return f"v:sha256:{h.hexdigest()}"

    @staticmethod
    def compute_asset_version(
        payload_bytes: Optional[bytes],
        code_fingerprint: str,
        prompt_fingerprint: Optional[str] = None,
        upstream_versions: Optional[List[str]] = None,
    ) -> str:
        """
        Compute deterministic content-addressed version for an asset.

        Includes:
        - Payload content
        - Code version (git commit, etc.)
        - Prompt/model version
        - Upstream asset versions

        Args:
            payload_bytes: Asset content bytes
            code_fingerprint: Code version (e.g., "git:abcd1234")
            prompt_fingerprint: Model/prompt version (e.g., "qwen3-vl:v7")
            upstream_versions: List of upstream asset versions

        Returns:
            Version string like "v:sha256:..."
        """
        h = hashlib.sha256()

        if payload_bytes:
            h.update(payload_bytes)

        h.update(code_fingerprint.encode())

        if prompt_fingerprint:
            h.update(prompt_fingerprint.encode())

        if upstream_versions:
            h.update("|".join(sorted(upstream_versions)).encode())

        return f"v:sha256:{h.hexdigest()}"

    def cleanup(self):
        """Cleanup resources."""
        self._db_executor.shutdown(wait=True)
