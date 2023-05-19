import asyncio
from typing import Any, Dict

from marie.executor.mixin import StorageMixin
from marie_server.scheduler.scheduler import Scheduler
from marie.logging.predefined import default_logger as logger

INIT_POLL_PERIOD = 0.250  # 250ms
MAX_POLL_PERIOD = 16.0  # 16s

class PsqlJobScheduler(Scheduler, StorageMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        print("config", config)
        self.task = None
        self.running = False

    def start_schedule(self) -> Any:
        logger.info("Starting scheduler")
        self.running = True
        self.task = asyncio.create_task(self.retrieve())

    async def retrieve(self):
        logger.info("Starting poller with psql")
        wait_time = INIT_POLL_PERIOD
        has_records = False
        while self.running:
            await asyncio.sleep(wait_time)
            logger.info(f"Polling for new jobs : {wait_time}")
            has_records = await self.get_records_for_run()
            wait_time = INIT_POLL_PERIOD if has_records else min(wait_time * 2, MAX_POLL_PERIOD)

    def stop_schedule(self) -> Any:
        pass

    def debug_info(self) -> str:
        pass

    async def get_records_for_run(self):
        pass
