from __future__ import annotations

from typing import List, Protocol

from marie.engine.llm_queue.models import QueueRequest
from marie.engine.llm_queue.result_types import BatchResult


class ExecutionAdapter(Protocol):
    def execute_requests(self, requests: List[QueueRequest]) -> List[BatchResult]: ...
