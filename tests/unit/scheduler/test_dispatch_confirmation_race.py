import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.scheduler.job_lock import AsyncJobLock
from marie.scheduler.models import WorkInfo
from marie.scheduler.psql import PostgreSQLJobScheduler
from marie.scheduler.services.attempt_lifecycle_service import AttemptLifecycleService
from marie.scheduler.state import WorkState


@dataclass(frozen=True)
class DispatchContractCase:
    name: str
    job_id: str
    run_attempt_id: str
    initial_stage: str
    event: str
    expected_outcome: str
    expected_attempt_state: str
    expected_capacity_action: str
    expected_reason: str | None = None


DISPATCH_CONTRACT_CASES = {
    'timeout_before_detached_send': DispatchContractCase(
        name='timeout_before_detached_send',
        job_id='06a63f90-0001-7000-8000-000000000001',
        run_attempt_id='10000000-0000-4000-8000-000000000001',
        initial_stage='pre_detach',
        event='dispatch_confirmation_timeout',
        expected_outcome='failed',
        expected_attempt_state='dispatch_failed',
        expected_capacity_action='scheduler_release',
        expected_reason='dispatch_timeout',
    ),
    'timeout_after_detached_send': DispatchContractCase(
        name='timeout_after_detached_send',
        job_id='06a63f90-0002-7000-8000-000000000002',
        run_attempt_id='10000000-0000-4000-8000-000000000002',
        initial_stage='post_detach_pre_rpc',
        event='dispatch_confirmation_timeout',
        expected_outcome='unknown',
        expected_attempt_state='dispatch_unknown',
        expected_capacity_action='scheduler_retain',
        expected_reason='dispatch_timeout',
    ),
    'unknown_later_admitted': DispatchContractCase(
        name='unknown_later_admitted',
        job_id='06a63f90-0003-7000-8000-000000000003',
        run_attempt_id='10000000-0000-4000-8000-000000000003',
        initial_stage='dispatch_unknown',
        event='admission_accepted',
        expected_outcome='confirmed',
        expected_attempt_state='dispatched',
        expected_capacity_action='retain_until_worker_adoption',
    ),
    'unknown_later_rejected_before_rpc': DispatchContractCase(
        name='unknown_later_rejected_before_rpc',
        job_id='06a63f90-0004-7000-8000-000000000004',
        run_attempt_id='10000000-0000-4000-8000-000000000004',
        initial_stage='dispatch_unknown',
        event='pre_rpc_rejection',
        expected_outcome='failed',
        expected_attempt_state='dispatch_failed',
        expected_capacity_action='scheduler_release',
        expected_reason='pre_send_rejected',
    ),
    'no_replicas': DispatchContractCase(
        name='no_replicas',
        job_id='06a63f90-0005-7000-8000-000000000005',
        run_attempt_id='10000000-0000-4000-8000-000000000005',
        initial_stage='post_detach_pre_rpc',
        event='replica_lookup_empty',
        expected_outcome='failed',
        expected_attempt_state='dispatch_failed',
        expected_capacity_action='scheduler_release',
        expected_reason='no_available_replicas',
    ),
    'send_crash_after_rpc_start': DispatchContractCase(
        name='send_crash_after_rpc_start',
        job_id='06a63f90-0006-7000-8000-000000000006',
        run_attempt_id='10000000-0000-4000-8000-000000000006',
        initial_stage='post_rpc',
        event='send_task_crashed',
        expected_outcome='unknown',
        expected_attempt_state='dispatch_unknown',
        expected_capacity_action='scheduler_retain',
        expected_reason='send_task_crashed',
    ),
    'recovered_attempt_late_worker_start': DispatchContractCase(
        name='recovered_attempt_late_worker_start',
        job_id='06a63f90-0007-7000-8000-000000000007',
        run_attempt_id='10000000-0000-4000-8000-000000000007',
        initial_stage='recovered',
        event='late_worker_start',
        expected_outcome='rejected',
        expected_attempt_state='recovered',
        expected_capacity_action='no_executor_invocation',
        expected_reason='stale_run_attempt',
    ),
}


@pytest.fixture
def dispatch_contract_cases() -> dict[str, DispatchContractCase]:
    cases = dict(DISPATCH_CONTRACT_CASES)
    assert len({case.job_id for case in cases.values()}) == len(cases)
    assert len({case.run_attempt_id for case in cases.values()}) == len(cases)
    return cases


class RaceRepository:
    def __init__(self) -> None:
        self.job_state = WorkState.ACTIVE
        self.dispatch_results: list[dict[str, Any]] = []
        self.fail_calls: list[dict[str, Any]] = []
        self.terminal_audits: list[dict[str, Any]] = []

    async def record_job_attempt_dispatch_started(self, **_fields: Any) -> None:
        return None

    async def record_job_attempt_dispatch_result(self, **fields: Any) -> None:
        self.dispatch_results.append(fields)

    async def fail_job(self, **fields: Any) -> tuple[int, None]:
        self.fail_calls.append(fields)
        if self.job_state == WorkState.COMPLETED:
            return 0, None
        raise AssertionError(f'Unexpected job state: {self.job_state}')

    async def record_job_attempt_terminal(self, **fields: Any) -> None:
        self.terminal_audits.append(fields)


class LateConfirmingJobManager:
    def __init__(self, repository: RaceRepository) -> None:
        self.repository = repository
        self.task: asyncio.Task[None] | None = None
        self.completed = asyncio.Event()

    async def submit_job(
        self,
        *,
        submission_id: str,
        confirmation_event: asyncio.Event,
        **_fields: Any,
    ) -> str:
        self.task = asyncio.create_task(
            self._complete_after_dispatch_timeout(confirmation_event)
        )
        return submission_id

    async def _complete_after_dispatch_timeout(
        self, confirmation_event: asyncio.Event
    ) -> None:
        await asyncio.sleep(0.2)
        confirmation_event.set()
        self.repository.job_state = WorkState.COMPLETED
        self.completed.set()


class RecordingSemaphoreStore:
    def __init__(self) -> None:
        self.releases: list[tuple[str, str, str]] = []

    def release_owned(self, executor: str, job_id: str, *, owner: str) -> bool:
        self.releases.append((executor, job_id, owner))
        return True


def build_work_item() -> WorkInfo:
    now = datetime.now(timezone.utc)
    return WorkInfo(
        id='06a63f82-cb47-72da-8000-b33e4ceba4a1',
        dag_id='06a63f82-cb47-72da-8000-b33e4ceba49d',
        name='patient_indexing',
        priority=0,
        data={
            'name': 'patient_indexing',
            'metadata': {
                'on': 'patient_indexing_executor://default',
            },
        },
        state=WorkState.ACTIVE,
        retry_limit=1,
        retry_delay=0,
        retry_backoff=False,
        start_after=now,
        expire_in_seconds=3600,
        keep_until=now + timedelta(days=1),
        dependencies=[],
        job_level=0,
    )


@pytest.mark.asyncio
async def test_late_confirmation_reproduces_false_dispatch_cleanup_race() -> None:
    repository = RaceRepository()
    job_manager = LateConfirmingJobManager(repository)
    semaphore_store = RecordingSemaphoreStore()
    frontier = SimpleNamespace(release_lease_local=AsyncMock())
    scheduler = object.__new__(PostgreSQLJobScheduler)
    scheduler.logger = MagicMock()
    scheduler.repository = repository
    scheduler.job_manager = job_manager
    scheduler.frontier = frontier
    scheduler._semaphore_store = semaphore_store
    scheduler.lease_ttl_seconds = 1
    scheduler.lease_owner = 'reproducer-scheduler'
    scheduler.gateway_instance_id = 'reproducer-gateway'
    scheduler.notify_event = AsyncMock(return_value=True)

    counters: list[tuple[str, dict[str, Any]]] = []
    scheduler.attempt_lifecycle_service = AttemptLifecycleService(
        repository=repository,
        frontier=frontier,
        dag_service=SimpleNamespace(),
        control_flow_service=SimpleNamespace(),
        status_update_lock=AsyncJobLock(),
        job_cache={},
        scheduler_lease_owner=scheduler.lease_owner,
        gateway_instance_id=scheduler.gateway_instance_id,
        notify_callback=scheduler.notify_event,
        counter_callback=lambda name, **fields: counters.append((name, fields)),
    )

    work_item = build_work_item()
    run_attempt_id = 'e72e442c-a836-45a4-b66d-a4a460c3b04e'

    enqueued, _ = await asyncio.gather(
        scheduler.enqueue(
            work_item,
            run_owner=scheduler.lease_owner,
            run_attempt_id=run_attempt_id,
        ),
        job_manager.completed.wait(),
    )

    # Freeze the current four-part race signature. The confirmation waiter times
    # out, detached work completes late, the bool result becomes the error text
    # "False", and cleanup loses its terminal transition to that late success.
    assert enqueued is False
    assert job_manager.task is not None
    assert job_manager.task.done()
    # This COMPLETED state is the late success interleaved with the timeout errors.
    assert repository.job_state == WorkState.COMPLETED

    await scheduler._handle_dispatch_failure(
        work_item,
        'patient_indexing_executor',
        work_item.id,
        enqueued,
        run_owner=scheduler.lease_owner,
        run_attempt_id=run_attempt_id,
    )

    assert repository.dispatch_results == [
        {
            'run_attempt_id': run_attempt_id,
            'confirmed': False,
            'error': 'dispatch_timeout',
        }
    ]
    assert repository.fail_calls[0]['output_metadata'] == {
        'dispatch_failed': True,
        'dispatch_error': 'False',
        'failure_stage': 'enqueue',
        'failure_source': 'dispatch_failure',
        'error_message': 'False',
    }
    assert repository.terminal_audits[0]['accepted'] is False
    assert repository.terminal_audits[0]['reject_reason'] == 'db_update_zero_rows'
    assert semaphore_store.releases == [
        ('patient_indexing_executor', work_item.id, work_item.id)
    ]
    scheduler.logger.error.assert_any_call(
        f'Timeout waiting for dispatch confirmation for job {work_item.id}'
    )
    scheduler.logger.error.assert_any_call(
        f'Dispatch failure cleanup could not transition job {work_item.id}; '
        f'run_attempt_id={run_attempt_id}'
    )
    assert counters[0][0] == 'terminal_event_stale_attempt_total'


TARGET_CONTRACT_PENDING = pytest.mark.xfail(
    strict=True,
    reason='Target dispatch lifecycle is implemented by Slices 02 through 07',
)


def _fail_pending_contract(case: DispatchContractCase) -> None:
    pytest.fail(
        f'{case.name}: {case.initial_stage} + {case.event} must produce '
        f'outcome={case.expected_outcome}, '
        f'attempt_state={case.expected_attempt_state}, '
        f'capacity={case.expected_capacity_action}, reason={case.expected_reason}'
    )


@TARGET_CONTRACT_PENDING
def test_timeout_before_detached_send_is_safe_failure(
    dispatch_contract_cases: dict[str, DispatchContractCase],
) -> None:
    _fail_pending_contract(dispatch_contract_cases['timeout_before_detached_send'])


@TARGET_CONTRACT_PENDING
def test_timeout_after_detached_send_is_unknown(
    dispatch_contract_cases: dict[str, DispatchContractCase],
) -> None:
    _fail_pending_contract(dispatch_contract_cases['timeout_after_detached_send'])


@TARGET_CONTRACT_PENDING
def test_unknown_later_admitted_becomes_dispatched(
    dispatch_contract_cases: dict[str, DispatchContractCase],
) -> None:
    _fail_pending_contract(dispatch_contract_cases['unknown_later_admitted'])


@TARGET_CONTRACT_PENDING
def test_unknown_later_rejected_before_rpc_becomes_failure(
    dispatch_contract_cases: dict[str, DispatchContractCase],
) -> None:
    _fail_pending_contract(
        dispatch_contract_cases['unknown_later_rejected_before_rpc']
    )


@TARGET_CONTRACT_PENDING
def test_no_replicas_rejects_before_confirmation_timeout(
    dispatch_contract_cases: dict[str, DispatchContractCase],
) -> None:
    _fail_pending_contract(dispatch_contract_cases['no_replicas'])


@TARGET_CONTRACT_PENDING
def test_send_crash_after_rpc_start_remains_unknown(
    dispatch_contract_cases: dict[str, DispatchContractCase],
) -> None:
    _fail_pending_contract(dispatch_contract_cases['send_crash_after_rpc_start'])


@TARGET_CONTRACT_PENDING
def test_recovered_attempt_rejects_late_worker_start(
    dispatch_contract_cases: dict[str, DispatchContractCase],
) -> None:
    _fail_pending_contract(
        dispatch_contract_cases['recovered_attempt_late_worker_start']
    )
