import pytest

from marie._docarray import DocumentArray, LegacyDocument, docarray_v2
from marie.job.common import JobInfo, JobStatus
from marie.logging_core.logger import MarieLogger
from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler


class RecordingJobInfoClient:
    def __init__(self):
        self.calls = []
        self.stored: dict[str, JobInfo] = {}

    async def put_status(self, job_id, status, **kwargs):
        job_info = JobInfo(
            status=status,
            entrypoint='test-worker',
            **kwargs['jobinfo_replace_kwargs'],
        )
        self.stored[job_id] = JobInfo.from_json(job_info.to_json())
        self.calls.append(
            {
                "job_id": job_id,
                "status": status,
                "kwargs": kwargs,
            }
        )


@pytest.mark.parametrize(
    'output',
    [
        {'games': [{'id': 'one', 'moves': ['e4']}]},
        [{'id': 'one', 'moves': ['e4']}],
        [],
    ],
)
async def test_success_persists_a_bounded_typed_output_for_workflow_handoff(output):
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MarieLogger('test-successful-output')
    handler._deployment = 'worker-one'
    handler._job_info_client = RecordingJobInfoClient()
    handler._sem_untrack = lambda *_args, **_kwargs: True
    handler._request_attributes = lambda _requests: {'source': 'test'}
    handler._schedule_deployment_ready_after_terminal = lambda **_kwargs: None
    handler.is_dry_run = lambda _requests: False
    await handler._record_successful_job('job-one', [], {}, return_data=output)
    result = handler._job_info_client.stored['job-one'].runtime_env['output']
    assert result == output


@pytest.mark.skipif(not docarray_v2, reason='LegacyDocument requires docarray v2')
async def test_success_persists_transport_documents_through_job_info_serialization() -> None:
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MarieLogger('test-document-output')
    handler._deployment = 'worker-one'
    handler._job_info_client = RecordingJobInfoClient()
    handler._sem_untrack = lambda *_args, **_kwargs: True
    handler._request_attributes = lambda _requests: {'source': 'test'}
    handler._schedule_deployment_ready_after_terminal = lambda **_kwargs: None
    handler.is_dry_run = lambda _requests: False
    docs = DocumentArray[LegacyDocument](
        [LegacyDocument(id='document-one', text='indexed document', tags={'chunks': 1})]
    )

    await handler._record_successful_job('job-one', [], {}, return_data=docs)

    job_info = handler._job_info_client.stored.get('job-one')
    assert job_info is not None
    assert job_info.status == JobStatus.SUCCEEDED
    assert job_info.runtime_env['attributes'] == {'source': 'test'}
    assert job_info.runtime_env['output'] == [
        {
            'id': 'document-one',
            'tensor': None,
            'blob': None,
            'text': 'indexed document',
            'url': None,
            'embedding': None,
            'tags': {'chunks': 1},
            'scores': None,
            'chunks': None,
            'matches': None,
        }
    ]


async def test_extraction_handoff_persists_the_reference_without_repeating_document_content():
    handler = object.__new__(WorkerRequestHandler)
    handler.logger = MarieLogger('test-extraction-output')
    handler._deployment = 'worker-one'
    handler._job_info_client = RecordingJobInfoClient()
    handler._sem_untrack = lambda *_args, **_kwargs: True
    handler._request_attributes = lambda _requests: {}
    handler._schedule_deployment_ready_after_terminal = lambda **_kwargs: None
    handler.is_dry_run = lambda _requests: False
    reference = {'kind': 'extraction', 'ref_id': 'output', 'ref_type': 'document', 'sha256': 'a' * 64}
    await handler._record_successful_job('job-one', [], {}, return_data={
        'status': 'succeeded', 'input_ref': reference, 'metadata': {'ocr': 'large saved output'},
    })
    output = handler._job_info_client.stored['job-one'].runtime_env['output']
    assert output == {'input_ref': reference}


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
