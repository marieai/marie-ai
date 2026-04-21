import asyncio

from marie.engine.batch_processor import BatchProcessor


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _build_processor(max_concurrency: int) -> BatchProcessor:
    processor = object.__new__(BatchProcessor)
    processor.client = None
    processor.model_string = "test-model"
    processor.logger = _Logger()
    processor.max_concurrency = max_concurrency
    processor.default_completion_params = {}
    processor._shared_request_semaphore = None
    processor._shared_request_semaphore_loop = None
    return processor


def test_load_batched_request_shares_concurrency_limit_across_overlapping_calls():
    processor = _build_processor(max_concurrency=2)
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def fake_completion(**kwargs):
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            task_id = kwargs["task_id"]
            return task_id, f"response:{task_id}"
        finally:
            async with lock:
                active -= 1

    processor.acompletion_with_retry = fake_completion

    async def run():
        results = await asyncio.gather(
            processor.load_batched_request(
                messages_list=[["a"], ["b"], ["c"]],
                request_id="req-1",
                guided_json=None,
            ),
            processor.load_batched_request(
                messages_list=[["d"], ["e"], ["f"]],
                request_id="req-2",
                guided_json=None,
            ),
        )

        assert len(results) == 2
        assert peak == 2

    asyncio.run(run())
