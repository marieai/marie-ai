from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest
from marie.instrumentation.context import inject_trace_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState

from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState
from marie.serve.executors import BaseExecutor
from marie.serve.executors.decorators import requests


class LinkedExecutor(BaseExecutor):
    @requests(on='/work')
    async def work(self, parameters: dict, **kwargs: object) -> dict:
        with self.tracer.start_as_current_span('batch'):
            return {'status': 'ok'}


@pytest.fixture
def telemetry() -> (
    Iterator[tuple[LinkedExecutor, InMemorySpanExporter, TracerProvider]]
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    executor = LinkedExecutor()
    executor.tracer = provider.get_tracer('test.execution')
    yield executor, exporter, provider
    executor.close()
    provider.shutdown()


async def test_delayed_execution_and_retry_link_to_submission_in_separate_traces(
    telemetry,
) -> None:
    executor, exporter, provider = telemetry
    tracer = provider.get_tracer('test.submission')
    distributor = GatewayJobDistributor()
    submissions = []
    executions = []
    for job_id in ('job-a', 'job-b'):
        with tracer.start_as_current_span('POST /api/v1/invoke') as submission:
            carrier = inject_trace_context()
            submissions.append(submission.get_span_context())
        tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
        job = WorkInfo(
            id=job_id,
            dag_id=f'dag-{job_id}',
            name='work',
            data={
                'metadata': {'uri': 's3://marie/extract/sample.tif'},
                'submission_trace_context': carrier,
            },
            state=WorkState.CREATED,
            retry_limit=1,
            retry_delay=1,
            retry_backoff=False,
            start_after=tomorrow,
            expire_in_seconds=60,
            keep_until=tomorrow + timedelta(days=1),
        )
        restored = WorkInfo.model_validate_json(job.model_dump_json())
        assert restored.start_after == tomorrow
        for attempt in ('first', 'retry'):
            job_info = JobInfo(
                status=JobStatus.PENDING,
                entrypoint='executor:///work',
                metadata={
                    **restored.data,
                    'dag_id': restored.dag_id,
                    'run_attempt_id': f'{job_id}-{attempt}',
                },
            )
            parameters, _ = await distributor._build_payload(job_id, job_info)
            result = await executor.__acall__(
                '/work', tracing_context=Context(), parameters=parameters
            )
            assert result == {'status': 'ok'}
        executions.extend(
            s
            for s in exporter.get_finished_spans()
            if s.name == 'LinkedExecutor.work'
            and s.attributes['marie.job_id'] == job_id
        )
    assert len(executions) == 4
    assert len({s.context.trace_id for s in executions}) == 4
    for index, execution in enumerate(executions):
        origin = submissions[index // 2]
        assert execution.parent is None
        assert execution.context.trace_id != origin.trace_id
        assert len(execution.links) == 1
        assert execution.links[0].context.trace_id == origin.trace_id
        assert execution.links[0].context.span_id == origin.span_id
        assert execution.links[0].attributes['marie.link.kind'] == 'submission'
        assert (
            execution.attributes['marie.submission_id']
            == f'dag-job-{"a" if index < 2 else "b"}'
        )
        batches = [
            s
            for s in exporter.get_finished_spans()
            if s.name == 'batch' and s.context.trace_id == execution.context.trace_id
        ]
        assert len(batches) == 1
        assert batches[0].parent.span_id == execution.context.span_id


async def test_submission_link_preserves_flags_and_vendor_state_without_becoming_parent(
    telemetry,
) -> None:
    executor, exporter, _ = telemetry
    origin = SpanContext(
        trace_id=1,
        span_id=2,
        is_remote=True,
        trace_flags=TraceFlags(0),
        trace_state=TraceState([('vendor', 'value')]),
    )
    with trace.use_span(NonRecordingSpan(origin)):
        carrier = inject_trace_context()
    await executor.__acall__(
        '/work',
        tracing_context=Context(),
        parameters={'submission_trace_context': carrier},
    )
    execution = next(
        s for s in exporter.get_finished_spans() if s.name == 'LinkedExecutor.work'
    )
    assert execution.parent is None
    link = execution.links[0]
    assert link.context.trace_flags == TraceFlags(0)
    assert link.context.trace_state == TraceState([('vendor', 'value')])


@pytest.mark.parametrize(
    'carrier',
    [
        None,
        {},
        'bad',
        [],
        {'traceparent': 'bad'},
        {'traceparent': 'x' * 513},
        {'traceparent': 123},
    ],
)
async def test_missing_or_invalid_submission_context_does_not_break_execution(
    telemetry,
    carrier,
) -> None:
    executor, exporter, _ = telemetry
    await executor.__acall__(
        '/work',
        tracing_context=Context(),
        parameters={'submission_trace_context': carrier},
    )
    execution = next(
        s for s in exporter.get_finished_spans() if s.name == 'LinkedExecutor.work'
    )
    assert not execution.links


async def test_submission_link_keeps_the_parent_from_the_current_execution(
    telemetry,
) -> None:
    executor, exporter, _ = telemetry
    incoming = SpanContext(
        trace_id=10, span_id=20, is_remote=True, trace_flags=TraceFlags(1)
    )
    carrier = {'traceparent': '00-00000000000000000000000000000001-0000000000000002-01'}
    await executor.__acall__(
        '/work',
        tracing_context=trace.set_span_in_context(
            NonRecordingSpan(incoming), Context()
        ),
        parameters={'submission_trace_context': carrier},
    )
    execution = next(
        s for s in exporter.get_finished_spans() if s.name == 'LinkedExecutor.work'
    )
    assert execution.context.trace_id == 10
    assert execution.parent.span_id == 20
    assert execution.links[0].context.trace_id == 1
    assert execution.links[0].context.span_id == 2
