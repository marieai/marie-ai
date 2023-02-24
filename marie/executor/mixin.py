from typing import Dict, Optional

from docarray import DocumentArray

from marie.excepts import BadConfigSource
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage
from marie.timer import Timer


class StorageMixin:
    """Storage mixing providing storage capabilities"""

    def setup_storage(
        self,
        storage_enabled: Optional[bool] = False,
        storage_conf: Dict[str, str] = None,
        silence_exceptions: bool = False,
    ) -> None:
        """
        Setup document storage

        :param storage_enabled:
        :param storage_conf:
        @param silence_exceptions:
        """
        self.storage_enabled = storage_enabled
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
            except Exception as e:
                if silence_exceptions:
                    self.logger.warning(
                        "Storage enabled but config not setup correctly", exc_info=1
                    )
                else:
                    raise BadConfigSource(
                        "Storage enabled but config not setup correctly"
                    ) from e

    @Timer(text="stored in {:.4f} seconds")
    def store(
        self, ref_id: str, ref_type: str, store_mode: str, docs: DocumentArray
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
            self.logger.error(f"Unable to store documents : {e}")
