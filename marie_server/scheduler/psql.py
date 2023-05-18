import asyncio
from typing import Any, Dict

from marie.executor.mixin import StorageMixin
from marie_server.scheduler.scheduler import Scheduler


class PsqlJobScheduler(Scheduler, StorageMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        print("config", config)
        self.task = None

    def start_schedule(self) -> Any:
        self.task = asyncio.create_task(self.retrieve())

    async def retrieve(self):
        print("Start retrieve")
        while True:
            print("Pulling from psql")
            await asyncio.sleep(0.2)

    def stop_schedule(self) -> Any:
        pass

    def debug_info(self) -> str:
        pass
