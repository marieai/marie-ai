from typing import Any, Dict

from marie.executor.mixin import StorageMixin
from marie_server.scheduler.scheduler import Scheduler


class PsqlJobScheduler(Scheduler, StorageMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        print("config", config)

    def start_schedule(self) -> Any:
        pass

    def stop_schedule(self) -> Any:
        pass

    def debug_info(self) -> str:
        pass
