"""Fixed S03/S12/S13 qualification on owned stores and an AIMock-backed listener.

Run with python -m tools.stress.llm_dispatch_failure_scenarios --help.
The listener injects fixed response faults after AIMock receipt; AIMock itself
receives valid synthetic completion requests and does not emit malformed HTTP.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import uuid4

import httpx
from marie.engine.completion_contract import CompletionCallParams
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.endpoint import ExecutionOutcome, RegisteredEndpoint
from marie.engine.llm_queue.producer import V3Producer
from marie.engine.llm_queue.request_dispatcher import DispatchLane, RequestDispatcher
from marie.engine.llm_queue.store import RequestStore, StoreLimits

from tools.stress.llm_v3_aimock_e2e import (
    QualificationError,
    eventually,
    inspect_owned_fixture,
    load_fixture_manifest,
    preflight_fixtures,
    record_failure,
    repair_report_value,
    require,
    source_manifest,
)

# category, remote settled, response fully consumed by the production transport
EXPECTED = {
    'S12-5xx': ('provider_5xx', False, False),
    'S12-read-timeout': ('read_timeout', False, False),
    'S12-interrupted-body': ('protocol_error', False, False),
    'S13-malformed-http-json': ('invalid_response', True, True),
    'S13-authentication': ('authentication', True, False),
    'S13-invalid-request': ('invalid_request', True, False),
    'S13-assistant-non-json': ('success', True, True),
}
CASES = ('S03', *EXPECTED)


class OwnedFailureListener:
    """An owned loopback listener with fixed faults and explicit receipt/release."""

    def __init__(self, case: str, forward: Callable[[dict], Awaitable[dict]]) -> None:
        require(case in CASES, 'fixed_case')
        self.case, self.forward = case, forward
        self.received, self.release, self.body_started = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        self.interrupt = asyncio.Event()
        self.records: list[dict] = []
        self.tasks: set[asyncio.Task] = set()
        self.errors: list[BaseException] = []
        self.server: asyncio.Server | None = None
        self.port = 0
        self.before_first_response: Callable[[], Awaitable[None]] | None = None

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.port}/v1'

    async def start(self) -> None:
        require(self.server is None or not self.server.is_serving(), 'listener_owned')
        self.server = await asyncio.start_server(self._handle, '127.0.0.1', self.port)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop_listener(self) -> None:
        require(self.server is not None and self.server.is_serving(), 'listener_owned')
        require(
            self.server.sockets[0].getsockname()[:2] == ('127.0.0.1', self.port),
            'listener_identity',
        )
        self.server.close()
        await self.server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            header = await reader.readuntil(b'\r\n\r\n')
            headers = dict(
                line.split(':', 1)
                for line in header.decode().split('\r\n')
                if ':' in line
            )
            length = int(
                next(v for k, v in headers.items() if k.lower() == 'content-length')
            )
            require(0 < length <= 16 * 1024**2, 'synthetic_body_size')
            body = json.loads(await reader.readexactly(length))
            response = await self.forward(body)
            fault = self.case != 'S03' and body['model'] == 'fault'
            row = dict(
                model=body['model'],
                aimock_accepted=True,
                request_bytes=length,
                receipt_before_release=not self.release.is_set(),
            )
            self.records.append(row)
            if self.case == 'S03':
                content = body['messages'][-1]['content']
                index = int(
                    content[-1]['text'] if isinstance(content, list) else content
                )
                response = {'choices': [{'message': {'content': f'output-{index}'}}]}
                row['index'] = index
                if index == 0 and self.before_first_response:
                    await self.before_first_response()
            if fault:
                self.received.set()
                await self.release.wait()
            status = 200
            payload = json.dumps(response).encode()
            if fault:
                if self.case == 'S12-read-timeout':
                    await self.interrupt.wait()
                    return
                if self.case == 'S12-5xx':
                    status = 503
                elif self.case == 'S13-authentication':
                    status = 401
                elif self.case == 'S13-invalid-request':
                    status = 422
                elif self.case == 'S13-malformed-http-json':
                    payload = b'{invalid'
            interrupted = fault and self.case == 'S12-interrupted-body'
            wire_length = len(payload) + 100 if interrupted else len(payload)
            writer.write(
                f'HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {wire_length}\r\n\r\n'.encode()
                + payload
            )
            await writer.drain()
            if interrupted:
                self.body_started.set()
                await self.interrupt.wait()
        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception as error:
            self.errors.append(error)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self.tasks.discard(task)

    async def close(self) -> None:
        self.release.set()
        self.interrupt.set()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def accept_report(report: dict) -> None:
    """Accept only the selected exact cells with measured counts and owned cleanup."""
    selected = report['selected']
    require(
        isinstance(selected, list)
        and bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(CASES),
        'selected_cases',
    )
    require(report['status'] == 'passed' and report['failures'] == [], 'run_status')
    require(
        set(report['verified_stores']) == {'redis', 'valkey'}
        and all(r.get('verified') is True for r in report['verified_stores'].values())
        and report['aimock'].get('verified') is True,
        'verified_fixtures',
    )
    require(
        set(report['source']) == {'backend', 'studio'}
        and all(len(r['digest']) == 64 for r in report['source'].values()),
        'source_manifests',
    )
    expected = {
        f'{family}-{case}' for family in ('redis', 'valkey') for case in selected
    }
    rows = report['scenarios']
    require(
        len(rows) == len(expected) and {r['name'] for r in rows} == expected,
        'exact_cells',
    )
    for row in rows:
        case = row['case']
        require(
            row['status'] == 'passed' and row['name'] == f"{row['family']}-{case}",
            'cell_identity',
        )
        counts = {
            'provider_acceptances': 9 if case == 'S03' else 2,
            'writes': 9 if case == 'S03' else 0,
            'attempts': 9 if case == 'S03' else 2,
            'sibling_completions': 0 if case == 'S03' else 1,
        }
        for field, value in counts.items():
            require(type(row[field]) is int and row[field] == value, 'cell_' + field)
        if case == 'S03':
            before, during, after = (
                row[k] for k in ('before_outage', 'during_outage', 'after_recovery')
            )
            for snapshot, count in ((before, 1), (after, 9)):
                require(
                    all(
                        type(snapshot[k]) is int and snapshot[k] == count
                        for k in ('writes', 'provider_acceptances')
                    ),
                    'outage_counts',
                )
            require(
                type(during['pending']) is int
                and during['pending'] == 8
                and during['caller_pending'] is True
                and during['circuit'] == 'open',
                'outage_pending',
            )
            require(
                before['first_output_sha256'] == after['first_output_sha256']
                and len(before['first_output_sha256']) == 64,
                'original_output_hash',
            )
            require(
                all(
                    row[k] is True
                    for k in (
                        'same_producer',
                        'same_deadlines',
                        'ordered_unique_outputs',
                    )
                ),
                'original_call',
            )
        else:
            category, settled, consumed = EXPECTED[case]
            require(
                row['category'] == category
                and row['remote_settled'] is settled
                and row['fully_consumed'] is consumed
                and row['transmitted'] is True,
                'transport_outcome',
            )
            for key, value in [
                ('reserved_before_close', int(not settled)),
                ('reserved_after_close', int(not settled)),
                ('execution_seq', 1),
                ('retries', 0),
            ]:
                require(type(row[key]) is int and row[key] == value, 'ownership_' + key)
            require(row['receipt_before_release'] is True, 'receipt_boundary')
            require(
                row['application_json_rejected'] is (case == 'S13-assistant-non-json'),
                'application_boundary',
            )
            state = (
                'outcome_unknown'
                if not settled
                else 'succeeded'
                if category == 'success'
                else 'failed'
            )
            require(
                row['state_before_close'] == state
                and row['same_producer'] is True
                and row['same_deadlines'] is True,
                'fault_ownership',
            )
            require(
                row['aimock_acceptances_by_item'] == {'fault': 1, 'sibling': 1}
                and all(
                    type(v) is int for v in row['aimock_acceptances_by_item'].values()
                ),
                'per_item_acceptances',
            )
    cleanup = report['cleanup']
    require(
        len(cleanup) == len(rows)
        and len({r['fabric'] for r in rows}) == len(rows)
        and {r['fabric'] for r in cleanup} == {r['fabric'] for r in rows},
        'cleanup_cells',
    )
    require(
        all(
            type(r['remaining_keys']) is int and r['remaining_keys'] == 0
            for r in cleanup
        ),
        'cleanup_keys',
    )


def finalize_report(report: dict, output: Path) -> None:
    """Persist failures with the baseline's content-free reporting and repair helpers."""
    try:
        accept_report(report)
    except Exception as error:
        if report['status'] != 'failed':
            record_failure(report, error, phase='acceptance')
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    path = output / f"{report['run_id']}.json"
    try:
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        )
    except Exception as error:
        record_failure(report, error, phase='report_write')
        try:
            repaired = repair_report_value(report)
        except Exception as secondary:
            record_failure(report, secondary, phase='report_repair')
            repaired = dict(
                schema_version=1,
                run_id=report['run_id'],
                status='failed',
                report_unavailable=True,
                failures=report['failures'],
            )
        report.clear()
        report.update(repaired)
        try:
            path.write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
            )
        except Exception as error:
            record_failure(report, error, phase='report_write')


async def qualify(args: argparse.Namespace) -> dict:
    report = dict(
        schema_version=1,
        run_id=str(uuid4()),
        status='incomplete',
        failures=[],
        selected=list(args.scenarios),
        scenarios=[],
        cleanup=[],
        source={},
        phase='manifest',
        scenario=None,
        started_at=datetime.now(timezone.utc).isoformat(),
        boundary='Owned loopback response seam after AIMock completion receipt; fixed bytes are injected by the seam',
    )
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        # The shared manifest guard retains its baseline item contract unchanged.
        args.items = 80
        fixtures = load_fixture_manifest(args)
        require(
            bool(args.scenarios)
            and len(set(args.scenarios)) == len(args.scenarios)
            and set(args.scenarios) <= set(CASES),
            'selected_cases',
        )
        report['phase'] = 'preflight'
        stores, mock, container = await asyncio.to_thread(
            preflight_fixtures, args, fixtures
        )
        report['verified_stores'], report['aimock'] = stores, mock
        for name, root in [
            ('backend', Path(__file__).resolve().parents[2]),
            ('studio', Path(args.studio)),
        ]:
            manifest = source_manifest(root)
            (args.output / f"{report['run_id']}-{name}-source.json").write_text(
                json.dumps(manifest, indent=2)
            )
            report['source'][name] = {
                'base_commit': manifest['base_commit'],
                'digest': manifest['canonical_path_content_sha256'],
            }
        async with httpx.AsyncClient(timeout=5, trust_env=False) as http:
            for family, fixture in fixtures.items():
                for case in args.scenarios:
                    report['phase'], report['scenario'] = 'workload', f'{family}-{case}'
                    await run_cell(args, report, family, fixture, mock, http, case)
        if not report['failures']:
            report['status'] = 'passed'
    except Exception as error:
        record_failure(
            report, error, phase=report['phase'], scenario=report['scenario']
        )
    finally:
        finalize_report(report, args.output)
    return report


async def run_cell(
    args: argparse.Namespace,
    report: dict,
    family: str,
    fixture: dict,
    mock: dict,
    http: httpx.AsyncClient,
    case: str,
) -> None:
    import pytest

    # Existing executor fixture is checked in and supplies assets/config only.
    tests = str(Path(__file__).resolve().parents[2] / 'packages/marie-engine/tests')
    if tests not in sys.path:
        sys.path.insert(0, tests)
    from test_v3_image_lifetime import executor_request
    from test_v3_producer import engine_for

    inspect_owned_fixture(fixture, running=True)
    mock_fixture = dict(
        fixture,
        service='aimock',
        container_id=mock['container_id'],
        image_id=mock['image_id'],
        bindings=mock['bindings'],
    )
    inspect_owned_fixture(mock_fixture, running=True)
    reset = await http.post(
        args.admin + '/fault-profile', json={'profile': 'normal', 'resetCounters': True}
    )
    reset.raise_for_status()
    require(reset.json()['requestCount'] == 0, 'counter_reset')

    async def forward(body: dict) -> dict:
        response = await http.post(args.mock + '/v1/chat/completions', json=body)
        response.raise_for_status()
        return response.json()

    seam = OwnedFailureListener(case, forward)
    row = dict(
        name=f'{family}-{case}',
        case=case,
        family=family,
        fabric='failure-' + uuid4().hex,
        status='incomplete',
        provider_acceptances=0,
        writes=0,
        attempts=0,
        sibling_completions=0,
    )
    report['scenarios'].append(row)
    producer = engine = store = runtime = None
    tasks = []
    patch = pytest.MonkeyPatch()
    admissions = {}
    try:
        await seam.start()
        store = RequestStore(
            fixture['url'],
            fabric_id=row['fabric'],
            version='v3',
            limits=StoreLimits(
                max_execution_items=2, remote_uncertainty_ms=100, delivery_grace_ms=1000
            ),
        )
        store.test_url = fixture['url']
        runtime = RequestDispatcher(
            store=store,
            endpoints=[
                RegisteredEndpoint(
                    'fixed',
                    seam.url,
                    allow_loopback=True,
                    execution_limit=1 if case == 'S03' else 2,
                    call_timeout_seconds=5 if case == 'S03' else 1,
                )
            ],
            lanes=[
                DispatchLane('pool', 'fixed', execution_limit=1 if case == 'S03' else 2)
            ],
            poll_seconds=0.01,
            retry_min_ms=100,
            retry_max_ms=100,
            circuit_open_ms=250,
            drain_seconds=0.2,
        )
        original_admit = RequestStore.admit

        def observed_admit(self: RequestStore, request: Any) -> Any:
            result = original_admit(self, request)
            if self.keys.fabric_id == store.keys.fabric_id:
                admissions[request.attempt_id] = request
            return result

        patch.setattr(RequestStore, 'admit', observed_admit)
        await runtime.start()
        await eventually(lambda: store.resolve_route('pool'))
        if case == 'S03':
            engine = engine_for(store, patch, multimodal=True, timeout=30)
            producer = engine.batch_processor._get_queued_executor()
            await run_recovery(
                args,
                row,
                store,
                runtime,
                seam,
                http,
                producer,
                engine,
                patch,
                admissions,
                tasks,
                executor_request,
                mock_fixture,
            )
        else:
            producer = V3Producer(
                config=LlmQueueConfig(
                    enabled=True,
                    queue_url=fixture['url'],
                    pool_id='pool',
                    fabric_group_id=store.keys.fabric_id,
                    queue_contract_version='v3',
                )
            )
            await run_failure(
                args, row, store, runtime, seam, http, producer, admissions, tasks
            )
        row['attempts'] = len(admissions)
        require(not seam.errors, 'listener_errors')
        row['status'] = 'passed'
    finally:
        if producer is not None:
            try:
                await asyncio.to_thread(engine.close if engine else producer.close)
            except Exception as error:
                record_failure(
                    report, error, phase='cleanup_producer', scenario=row['name']
                )
        seam.release.set()
        seam.interrupt.set()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if runtime is not None:
            try:
                await runtime.stop()
            except Exception as error:
                record_failure(
                    report, error, phase='cleanup_runtime', scenario=row['name']
                )
        try:
            await seam.close()
        except Exception as error:
            record_failure(
                report, error, phase='cleanup_listener', scenario=row['name']
            )
        patch.undo()
        cleanup = dict(fabric=row['fabric'], remaining_keys=None)
        report['cleanup'].append(cleanup)
        if store is not None:
            try:
                # Only this cell's namespace, after ownership evidence is saved.
                keys = list(store.client.scan_iter(match=store.keys.prefix + '*'))
                if keys:
                    store.client.delete(*keys)
                cleanup['remaining_keys'] = len(
                    list(store.client.scan_iter(match=store.keys.prefix + '*'))
                )
            except Exception as error:
                record_failure(
                    report, error, phase='cleanup_store', scenario=row['name']
                )
            finally:
                try:
                    store.close()
                except Exception as error:
                    record_failure(
                        report, error, phase='cleanup_store_close', scenario=row['name']
                    )


async def provider_count(http: httpx.AsyncClient, args: argparse.Namespace) -> int:
    response = await http.get(args.admin + '/fault-profile')
    response.raise_for_status()
    count = response.json()['requestCount']
    require(type(count) is int, 'provider_integer')
    return count


async def run_recovery(
    args: argparse.Namespace,
    row: dict,
    store: RequestStore,
    runtime: RequestDispatcher,
    seam: OwnedFailureListener,
    http: httpx.AsyncClient,
    producer: V3Producer,
    engine: Any,
    patch: Any,
    admissions: dict,
    tasks: list[asyncio.Task],
    executor_request: Callable,
    mock_fixture: dict,
) -> None:
    from marie.extract.annotators import util

    work = args.output / row['fabric']
    work.mkdir()
    request, refs, phases = executor_request(work, patch, engine)
    output = work / 'agent-output/test'
    writes = []
    original_write = util._write_single_result

    def write(**kwargs: Any) -> Any:
        result = original_write(**kwargs)
        writes.append(kwargs['task_id'])
        return result

    patch.setattr(util, '_write_single_result', write)

    async def gate(opened: bool) -> None:
        await asyncio.to_thread(
            store.configure_endpoint,
            runtime.owner,
            'fixed',
            execution_limit=1,
            execution_bytes=64 * 1024**2,
            gate_open=opened,
        )

    seam.before_first_response = lambda: gate(False)
    task = asyncio.create_task(request)
    tasks.append(task)
    await eventually(
        lambda: (
            len(writes) == 1 and len(producer._pending) == 8 and len(admissions) == 9
        ),
        seconds=15,
    )
    first = output / 'page_0.md'
    before_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    original_producer = producer.producer_id
    original_deadlines = {
        key: request.expires_at_ms for key, request in admissions.items()
    }
    row['before_outage'] = dict(
        writes=len(writes),
        provider_acceptances=await provider_count(http, args),
        first_output_sha256=before_hash,
    )
    require(row['before_outage']['provider_acceptances'] == 1, 'first_receipt_count')
    require(len(seam.records) == 1 and seam.records[0]['index'] == 0, 'first_index')
    # Fault only the exact owned listener after its original output was written.
    inspect_owned_fixture(mock_fixture, running=True)
    await seam.stop_listener()
    await gate(True)
    await eventually(lambda: store.endpoint_status('fixed')['circuit'] == 'open')
    row['during_outage'] = dict(
        pending=len(producer._pending),
        caller_pending=not task.done(),
        circuit=store.endpoint_status('fixed')['circuit'],
        category=store.endpoint_status('fixed')['category'],
    )
    await gate(False)
    await eventually(lambda: not runtime._calling)
    await seam.start()
    await gate(True)
    result = await task
    require(result['status'] == 'success', 'executor_result')
    row['same_producer'] = producer.producer_id == original_producer and all(
        store.metadata(key).producer_id == original_producer for key in admissions
    )
    row['same_deadlines'] = original_deadlines == {
        key: store.metadata(key).expires_at_ms for key in admissions
    }
    outputs = [(output / f'page_{index}.md').read_text() for index in range(9)]
    row['ordered_unique_outputs'] = (
        outputs == [f'output-{index}' for index in range(9)]
        and len(writes) == len(set(writes)) == 9
    )
    row['writes'] = len(writes)
    row['provider_acceptances'] = await provider_count(http, args)
    row['after_recovery'] = dict(
        writes=len(writes),
        provider_acceptances=row['provider_acceptances'],
        first_output_sha256=hashlib.sha256(first.read_bytes()).hexdigest(),
    )
    row['provider_acceptance_epochs'] = [
        dict(
            before=0,
            at_outage=row['before_outage']['provider_acceptances'],
            after=row['provider_acceptances'],
        )
    ]
    row['output_sha256'] = [
        hashlib.sha256(value.encode()).hexdigest() for value in outputs
    ]
    row['executor_boundary'] = (
        'actual annotator_llm request and annotation asset callback'
    )
    row['annotation_asset_callbacks'] = sum(
        p['phase'] == 'annotation_assets' for p in phases
    )
    require(row['annotation_asset_callbacks'] == 1, 'annotation_assets')


async def run_failure(
    args: argparse.Namespace,
    row: dict,
    store: RequestStore,
    runtime: RequestDispatcher,
    seam: OwnedFailureListener,
    http: httpx.AsyncClient,
    producer: V3Producer,
    admissions: dict,
    tasks: list[asyncio.Task],
) -> None:
    outcomes = []
    consumed = []
    client = runtime._clients['fixed']
    original = client.execute
    if row['case'] == 'S12-read-timeout':
        client.client.timeout = httpx.Timeout(0.25)
        row['httpx_read_timeout_seconds'] = 0.25
        row['outer_call_timeout_seconds'] = 1

    async def observe_response(response: httpx.Response) -> None:
        marker = {'fully_consumed': False}
        consumed.append(marker)
        iterate = response.aiter_bytes

        async def observed(*args: Any, **kwargs: Any) -> AsyncIterator[bytes]:
            async for chunk in iterate(*args, **kwargs):
                yield chunk
            marker['fully_consumed'] = True

        response.aiter_bytes = observed

    client.client.event_hooks['response'].append(observe_response)

    async def execute(
        call: CompletionCallParams, *, timeout_seconds: float
    ) -> ExecutionOutcome:
        result = await original(call, timeout_seconds=timeout_seconds)
        outcomes.append((call.model, result))
        return result

    client.execute = execute
    fault = asyncio.create_task(
        asyncio.to_thread(
            producer.execute,
            calls=[
                CompletionCallParams(
                    model='fault', messages=[{'role': 'user', 'content': 'synthetic'}]
                )
            ],
            batch_request_id='fault',
            batch_timeout=10,
        )
    )
    tasks.append(fault)
    await asyncio.wait_for(seam.received.wait(), 5)
    require(await provider_count(http, args) == 1, 'fault_receipt')
    require(not fault.done(), 'fault_pending_at_receipt')
    sibling = asyncio.create_task(
        asyncio.to_thread(
            producer.execute,
            calls=[
                CompletionCallParams(
                    model='sibling', messages=[{'role': 'user', 'content': 'synthetic'}]
                )
            ],
            batch_request_id='sibling',
            batch_timeout=10,
        )
    )
    tasks.append(sibling)
    completed = await sibling
    row['sibling_completions'] = sum(result.error is None for result in completed)
    require(not fault.done(), 'sibling_independence')
    seam.release.set()
    if row['case'] == 'S12-interrupted-body':
        await asyncio.wait_for(seam.body_started.wait(), 2)
        require(seam.records[0]['aimock_accepted'], 'receipt_before_interrupt')
        seam.interrupt.set()
    await eventually(lambda: any(model == 'fault' for model, _ in outcomes))
    outcome = next(result for model, result in outcomes if model == 'fault')
    attempt = next(
        key for key, request in admissions.items() if request.call.model == 'fault'
    )
    await eventually(
        lambda: (
            store.metadata(attempt).state in {'succeeded', 'failed', 'outcome_unknown'}
        )
    )
    metadata = store.metadata(attempt)
    row.update(
        category=outcome.category or 'success',
        remote_settled=outcome.remote_settled,
        fully_consumed=(
            consumed[-1]['fully_consumed']
            if row['case'] != 'S12-read-timeout'
            else False
        ),
        transmitted=bool(seam.records[0]['aimock_accepted']),
        execution_seq=metadata.execution_seq,
        retries=runtime._counts['retries'],
        reserved_before_close=store.usage()['reserved_items'],
        receipt_before_release=seam.records[0]['receipt_before_release'],
        application_json_rejected=False,
        state_before_close=metadata.state,
    )
    if outcome.remote_settled:
        results = await fault
        if row['case'] == 'S13-assistant-non-json':
            from marie.extract.annotators import util
            from marie.extract.annotators.util import JSONOutputParserError

            work = args.output / row['fabric']
            work.mkdir()
            try:
                util._write_single_result(
                    None,
                    'synthetic',
                    'synthetic.png',
                    'fault',
                    results[0].response,
                    str(work),
                    'json',
                )
            except JSONOutputParserError:
                row['application_json_rejected'] = True
            require(
                not (work / 'synthetic.json').exists(), 'application_no_json_output'
            )
        else:
            require(results[0].error is not None, 'permanent_delivery_error')
    else:
        # Observe the actual uncertainty boundary, then prove it cannot settle capacity.
        await eventually(lambda: store.server_time_ms() > metadata.uncertainty_until)
        require(store.usage()['reserved_items'] == 1, 'uncertainty_is_not_settlement')
        require(not fault.done(), 'unknown_caller_pending')
    row['same_producer'] = all(
        store.metadata(key).producer_id == producer.producer_id for key in admissions
    )
    row['same_deadlines'] = all(
        store.metadata(key).expires_at_ms == request.expires_at_ms
        for key, request in admissions.items()
    )
    await asyncio.to_thread(producer.close)
    await asyncio.gather(fault, return_exceptions=True)
    row['reserved_after_close'] = store.usage()['reserved_items']
    row['provider_acceptances'] = await provider_count(http, args)
    require(len(seam.records) == 2, 'no_resend')
    row['aimock_acceptances_by_item'] = {
        model: sum(r['model'] == model for r in seam.records)
        for model in ('fault', 'sibling')
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stores', 'studio', 'mock', 'admin', 'aimock-container'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenarios', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args()
    report = asyncio.run(qualify(args))
    print(json.dumps({'run_id': report['run_id'], 'status': report['status']}))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
