import asyncio
import threading
import traceback
from typing import Any, Dict, List

from marie.executor.mixin import StorageMixin
from marie_server.scheduler.scheduler import Scheduler
from marie.logging.predefined import default_logger as logger

INIT_POLL_PERIOD = 1.250  # 250ms
MAX_POLL_PERIOD = 16.0  # 16s


class PsqlJobScheduler(Scheduler, StorageMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        print("config", config)
        self.task = None
        self.running = False

    def start_schedule(self) -> None:
        logger.info("Starting job scheduling agent")

        def _run():
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is None:
                    asyncio.run(self.__poll())
                else:
                    loop.run_until_complete(self.__poll())
            except Exception as e:
                logger.error(f"Unable to setup job scheduler: {e}")
                logger.error(traceback.format_exc())

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    async def __poll(self):
        print("Starting poller with psql")
        self.running = True
        wait_time = INIT_POLL_PERIOD

        has_records = False
        while self.running:
            print(f"Polling for new jobs : {wait_time}")
            await asyncio.sleep(wait_time)
            records = await self.get_records_for_run()
            has_records = len(records) > 0
            for record in records:
                print("record", record)
                await self.schedule(record)
            wait_time = (
                INIT_POLL_PERIOD if has_records else min(wait_time * 2, MAX_POLL_PERIOD)
            )

    def stop_schedule(self) -> None:
        pass

    def debug_info(self) -> str:
        pass

    async def get_records_for_run(self) -> List[Dict[str, Any]]:
        records = []
        records.append({"id": 1, "name": "test"})
        return records

    async def schedule(self, record):
        print("scheduling : ", record)
