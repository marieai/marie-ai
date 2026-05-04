from __future__ import annotations

import asyncio
import uuid
from typing import List

from marie.engine.async_helper import run_coroutine_in_current_loop
from marie.engine.batch_processor import BatchProcessor
from marie.engine.llm_queue.models import QueueRequest
from marie.engine.llm_queue.result_types import BatchResult


class LiteLlmExecutionAdapter:
    def __init__(self, batch_processor: BatchProcessor):
        self.batch_processor = batch_processor

    def execute_requests(self, requests: List[QueueRequest]) -> List[BatchResult]:
        if not requests:
            return []

        completion_params = requests[0].completion_params
        metadata_list = [request.metadata for request in requests]
        messages_list = [request.messages for request in requests]
        dispatcher_request_id = f"llm-queue-dispatch-{uuid.uuid4()}"
        timeout_candidates = [
            request.timeout_seconds
            for request in requests
            if request.timeout_seconds is not None
        ]
        timeout_budget = (
            min(timeout_candidates)
            if timeout_candidates
            else self.batch_processor.batch_timeout
        )
        return run_coroutine_in_current_loop(
            asyncio.wait_for(
                self.batch_processor.load_batched_request(
                    messages_list=messages_list,
                    request_id=dispatcher_request_id,
                    guided_json=None,
                    completion_params=completion_params,
                    metadata_list=metadata_list,
                ),
                timeout=timeout_budget,
            )
        )
