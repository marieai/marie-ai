import pytest

from marie.job.common import JobStatus
from marie.logging_core.logger import MarieLogger
from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler


class RecordingJobInfoClient:
    def __init__(self):
        self.calls = []

    async def put_status(self, job_id, status, **kwargs):
        self.calls.append(
            {
                "job_id": job_id,
                "status": status,
                "kwargs": kwargs,
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("return_data", "expected_type", "expected_message"),
    [
        (
            None,
            "RuntimeError",
            "Internal Server Error - request handler failed",
        ),
        (
            {
                "status": "error",
                "error": ("Batch inference failed",),
                "error_details": {
                    "type": "ContextWindowExceededError",
                    "message": "maximum context length is 42768 tokens",
                },
            },
            "ContextWindowExceededError",
            "maximum context length is 42768 tokens",
        ),
        (
            {"status": "error", "error": ("legacy executor failure",)},
            "RuntimeError",
            "legacy executor failure",
        ),
    ],
)
async def test_record_failed_job_without_exception_publishes_error_details(
    return_data, expected_type, expected_message
):
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MarieLogger("test-worker-request-handler-failure-reporting")
    handler._deployment = "worker-1"
    handler._job_info_client = RecordingJobInfoClient()
    handler._worker_state = None
    handler._set_deployment_status = lambda *_args, **_kwargs: None
    handler._sem_untrack = lambda *_args, **_kwargs: True
    handler._request_attributes = lambda _requests: {"source": "test"}
    handler._schedule_deployment_ready_after_terminal = lambda **_kwargs: None
    handler.is_dry_run = lambda _requests: False

    await handler._record_failed_job(
        job_id="job-1",
        requests=[],
        e=None,
        metadata_attributes={"client_disconnected": False},
        return_data=return_data,
    )

    assert len(handler._job_info_client.calls) == 1
    call = handler._job_info_client.calls[0]
    assert call["job_id"] == "job-1"
    assert call["status"] == JobStatus.FAILED
    runtime_env = call["kwargs"]["jobinfo_replace_kwargs"]["runtime_env"]
    assert runtime_env["attributes"]["source"] == "test"
    assert runtime_env["attributes"]["client_disconnected"] is False
    assert runtime_env["error"]["type"] == expected_type
    assert runtime_env["error"]["message"] == expected_message
    assert runtime_env["error"]["filename"] == "unknown"
