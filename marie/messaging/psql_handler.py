from typing import Any, List

from docarray import DocumentArray, Document

from marie.logging.logger import MarieLogger
from marie.messaging.toast_handler import ToastHandler
from marie.executor.mixin import StorageMixin


class PsqlToastHandler(ToastHandler, StorageMixin):
    """
    PSQL Toast Handler that writes events using JSONB format to a postgres database
    utilizing the marie Document Storage API
    """

    def __init__(self, config: Any, **kwargs: Any):
        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info("Initializing PSQL Toast Handler")
        print(config)

        if not config:
            self.storage_enabled = False
            self.logger.warning("Storage config not set - storage disabled")
            return
        self.storage_enabled = config.get("enabled", False)
        self.setup_storage(self.storage_enabled, config.get("storage_config", {}))

    def get_supported_events(self) -> List[str]:
        return ["*"]

    async def notify(self, notification: Any, **kwargs: Any) -> bool:
        print("PSQL")
        print(notification)
        if not self.storage_enabled:
            return False

        self.persist(
            ref_id=notification.get("jobid", None),
            ref_type=notification.get("event", "NA"),
            results=notification,
        )

    def persist(self, ref_id: str, ref_type: str, results: Any) -> None:
        """
        Persist results to storage backend
        :param ref_id:
        :param ref_type:
        :param results:
        :return:
        """
        if self.storage_enabled:
            docs = DocumentArray(
                [
                    Document(
                        content=results,
                        tags={
                            "action": "job",
                            "ttl": 48 * 60,
                        },
                    )
                ]
            )

            self.store(
                ref_id=ref_id,
                ref_type=ref_type,
                store_mode="content",
                docs=docs,
            )
