from typing import Any, Dict


class SyncManager:
    """
    SyncManager is responsible for synchronizing the state of the scheduler with the state of the executor if they are out of sync.
    We will also publish events to the event publisher to notify the user of the state of the Job

    Example : If we restart the scheduler, we need to synchronize the state of the executor jobs that have completed while the scheduler was down.

    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
