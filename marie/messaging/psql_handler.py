import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from docarray import DocList

from marie.api.docs import StorageDoc
from marie.excepts import BadConfigSource
from marie.executor.mixin import StorageMixin
from marie.helper import get_or_reuse_loop
from marie.logging_core.logger import MarieLogger
from marie.messaging.events import EventMessage
from marie.messaging.toast_handler import ToastHandler


class PsqlToastHandler(ToastHandler, StorageMixin):
    """
    PSQL Toast Handler that writes events using JSONB format to a postgres database
    utilizing the marie Document Storage API

    """

    def __init__(self, config: Any, **kwargs: Any):
        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info("Initializing PSQL Toast Handler")

        if not config:
            self.storage_enabled = False
            self.logger.warning("Storage config not set - storage disabled")
            return
        self.storage_enabled = config.get("enabled", False)
        self.setup_storage(self.storage_enabled, config)

        # our db driver is sync so we we will execute in another thread until we update
        self._loop = get_or_reuse_loop()
        self._db_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="db-executor-toast"
        )

    def get_supported_events(self) -> List[str]:
        return ["*"]

    async def __notify_task(
        self,
        notification: EventMessage,
        silence_exceptions: bool = False,
        **kwargs: Any
    ) -> None:
        try:
            if not self.storage_enabled:
                return
            await self.persist(
                ref_id=notification.jobid,
                ref_type=notification.event if notification.event else "NA",
                results=notification,
            )
        except Exception as e:
            if silence_exceptions:
                self.logger.warning(
                    "Toast enabled but config not setup correctly", exc_info=1
                )
            else:
                raise BadConfigSource(
                    "Toast enabled but config not setup correctly"
                ) from e

    async def notify(self, notification: EventMessage, **kwargs: Any) -> bool:
        if not self.storage_enabled:
            return False

        await self.__notify_task(notification, True, **kwargs)
        return True

    async def persist(self, ref_id: str, ref_type: str, results: Any) -> None:
        """
        Persist results to storage backend
        :param ref_id:
        :param ref_type:
        :param results:
        :return:
        """
        if self.storage_enabled:
            docs = DocList[StorageDoc](
                [
                    StorageDoc(
                        content=results,
                        tags={
                            "action": "job",
                            "ttl": 48 * 60,
                        },
                    )
                ]
            )

            store_func = functools.partial(
                self.store,
                ref_id=ref_id,
                ref_type=ref_type,
                store_mode="content",
                docs=docs,
            )

            await self._loop.run_in_executor(self._db_executor, store_func)
