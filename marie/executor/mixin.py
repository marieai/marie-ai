import traceback
from typing import Dict, List, Optional, Tuple

from docarray import DocList

from marie.api.docs import StorageDoc
from marie.excepts import BadConfigSource
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage


class StorageMixin:
    """Storage mixin providing storage and asset tracking capabilities"""

    def setup_storage(
        self,
        storage_enabled: Optional[bool] = False,
        storage_conf: Dict[str, str] = None,
        silence_exceptions: bool = False,
        asset_tracking_enabled: Optional[bool] = False,
    ) -> None:
        """
        Setup document storage and asset tracking

        :param storage_enabled: Enable document storage
        :param storage_conf: Storage configuration dict
        :param silence_exceptions: Silence storage setup exceptions
        :param asset_tracking_enabled: Enable asset tracking (default: False)
        """
        self.storage_enabled = storage_enabled
        self.asset_tracking_enabled = asset_tracking_enabled
        self.storage_conf = storage_conf
        self.storage_handler = None

        if storage_enabled:
            try:
                self.storage = PostgreSQLStorage(
                    hostname=storage_conf["hostname"],
                    port=int(storage_conf["port"]),
                    username=storage_conf["username"],
                    password=storage_conf["password"],
                    database=storage_conf["database"],
                    table=storage_conf["default_table"],
                )
                self.storage_handler = self.storage
            except Exception as e:
                if silence_exceptions:
                    self.logger.warning(
                        "Storage enabled but config not setup correctly", exc_info=1
                    )
                else:
                    raise BadConfigSource(
                        "Storage enabled but config not setup correctly"
                    ) from e

        # Initialize asset tracker if enabled
        if asset_tracking_enabled and storage_enabled:
            try:
                from marie.assets import AssetTracker

                self.asset_tracker = AssetTracker(
                    storage_handler=self.storage_handler,
                    storage_conf=storage_conf,
                )
                self.logger.info("Asset tracking enabled")
            except Exception as e:
                if silence_exceptions:
                    self.logger.warning(
                        "Asset tracking enabled but initialization failed", exc_info=1
                    )
                    self.asset_tracking_enabled = False
                else:
                    raise BadConfigSource(
                        "Asset tracking enabled but initialization failed"
                    ) from e
        elif asset_tracking_enabled and not storage_enabled:
            self.logger.warning(
                "Asset tracking requires storage_enabled=True. Disabling asset tracking."
            )
            self.asset_tracking_enabled = False

    # @Timer(text="stored in {:.4f} seconds")
    def store(
        self, ref_id: str, ref_type: str, store_mode: str, docs: DocList[StorageDoc]
    ) -> None:
        """Store results in configured storage provider
        EXAMPLE USAGE

        .. code-block:: python
           def __init__(
               self,
               model_name_or_path: Optional[Union[str, os.PathLike]] = None,
               storage_enabled: bool = False,
               storage_conf: Dict[str, str] = None,
               **kwargs,
           ):
               super().__init__(**kwargs)

               self.logger.info(f"Storage enabled: {storage_enabled}")
               self.setup_storage(storage_enabled, storage_conf)


           def _tags(index: int, ftype: str, checksum: str):
               return {
                   "index": index,
                   "type": ftype,
                   "ttl": 48 * 60,
                   "checksum": checksum,
               }


           if self.storage_enabled:
               frame_checksum = hash_frames_fast(frames=[frame])
               docs = DocumentArray(
                   [
                       Document(
                           blob=convert_to_bytes(real),
                           tags=_tags(i, "real", frame_checksum),
                       ),
                   ]
               )

               self.store(
                   ref_id=ref_id,
                   ref_type=ref_type,
                   store_mode="blob",
                   docs=docs,
               )


        :param ref_id:
        :param ref_type:
        :param store_mode:
        :param docs:
        """
        try:
            if self.storage_enabled and self.storage is not None:
                self.storage.add(
                    docs, store_mode, {"ref_id": ref_id, "ref_type": ref_type}
                )
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Unable to store documents : {e}")

    # Asset tracking helper methods
    def _get_upstream_versions(
        self, dag_id: Optional[str], node_task_id: Optional[str]
    ) -> List[str]:
        """
        Get versions of upstream assets for version computation.

        :param dag_id: DAG ID
        :param node_task_id: Node task ID
        :return: List of upstream asset versions
        """
        if not dag_id or not node_task_id or not self.asset_tracking_enabled:
            return []

        try:
            from marie.assets import DAGAssetMapper

            upstream = DAGAssetMapper.get_upstream_assets_for_node(
                dag_id=dag_id,
                node_task_id=node_task_id,
                get_connection_fn=self.storage_handler._get_connection,
                close_connection_fn=self.storage_handler._close_connection,
            )

            return [u["latest_version"] for u in upstream if u["latest_version"]]
        except Exception as e:
            self.logger.warning(f"Failed to get upstream versions: {e}")
            return []

    def _get_upstream_asset_tuples(
        self, dag_id: Optional[str], node_task_id: Optional[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Get upstream asset tuples for lineage recording.

        :param dag_id: DAG ID
        :param node_task_id: Node task ID
        :return: List of (asset_key, latest_version, partition_key) tuples
        """
        if not dag_id or not node_task_id or not self.asset_tracking_enabled:
            return []

        try:
            from marie.assets import DAGAssetMapper

            upstream = DAGAssetMapper.get_upstream_assets_for_node(
                dag_id=dag_id,
                node_task_id=node_task_id,
                get_connection_fn=self.storage_handler._get_connection,
                close_connection_fn=self.storage_handler._close_connection,
            )

            return [
                (u["asset_key"], u["latest_version"], u["partition_key"])
                for u in upstream
            ]
        except Exception as e:
            self.logger.warning(f"Failed to get upstream asset tuples: {e}")
            return []
