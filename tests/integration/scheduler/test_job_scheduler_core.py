import asyncio
from datetime import datetime
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel
from uuid_extensions import uuid7str

from marie.job.common import JobStatus
from marie.job.job_manager import JobManager
from marie.scheduler import PostgreSQLJobScheduler
from marie.scheduler.job_scheduler import JobScheduler
from marie.scheduler.models import WorkInfo
from marie.scheduler.state import WorkState
from marie.storage.kv.in_memory import InMemoryKV
from tests.core.test_job_manager import NoopJobDistributor
from tests.core.test_utils import async_delay, async_wait_for_condition_async_predicate

# Job Scheduler Tests are very similar to Job Manager Tests


def compare_pydantic_models(
        model1: BaseModel, model2: BaseModel, excludes: List
) -> Dict[str, Any]:
    """Compare two Pydantic models and return the differences.
    :param model1: model1 to compare
    :param model2: model2 to compare
    :param excludes:
    """
    if excludes is None:
        excludes = []
    dict1 = model1.dict()
    dict2 = model2.dict()

    differences = {}
    for key in dict1.keys() | dict2.keys():
        if key in excludes:
            continue
        if dict1.get(key) != dict2.get(key):
            differences[key] = {"model1": dict1.get(key), "model2": dict2.get(key)}

    return differences


async def check_job_scheduler_succeeded(
        job_scheduler: JobScheduler, job_id: str
) -> bool:
    data = await job_scheduler.get_job(job_id)
    status = data._worker_state
    if status == WorkState.FAILED:
        raise RuntimeError(f"Job failed! {data}")
    assert status in {WorkState.CREATED, WorkState.COMPLETED, WorkState.ACTIVE}
    return status == WorkState.COMPLETED


async def update_job_scheduler_status(
        job_scheduler: JobScheduler, job_id: str, job_status: WorkState
) -> None:
    await job_scheduler.put_status(job_id, job_status)


@pytest.fixture
def num_jobs(request):
    print("request.param", request.param)
    return request.param


def build_work_item(name: str, job_id: str = None) -> WorkInfo:
    if job_id is None:
        job_id = uuid7str()
    return WorkInfo(
        id=job_id,
        name=name,
        priority=0,
        data={},
        state=WorkState.CREATED,
        retry_limit=0,
        retry_delay=0,
        retry_backoff=False,
        start_after=datetime.now(),
        expire_in_seconds=0,
        keep_until=datetime.now(),
        on_complete=False,
    )


@pytest.mark.asyncio
@pytest.fixture
async def job_manager(tmp_path):
    storage = InMemoryKV()
    # TODO: Externalize the storage configuration
    storage_config = {
        "hostname": "127.0.0.1",
        "port": 5432,
        "username": "postgres",
        "password": "123456",
        "database": "postgres",
        "default_table": "kv_store_a",
        "max_pool_size": 5,
        "max_connections": 5,
    }

    # storage = PostgreSQLKV(config=storage_config, reset=True)
    yield JobManager(storage=storage, job_distributor=NoopJobDistributor())


@pytest.mark.asyncio
@pytest.fixture
async def job_scheduler(tmp_path, job_manager: JobManager):
    scheduler_config = {
        "hostname": "localhost",
        "port": 5432,
        "database": "postgres",
        "username": "postgres",
        "password": "123456",
    }
    print("job_manager", job_manager)
    JobManager.SLOTS_AVAILABLE = 2
    print("SLOTS_AVAILABLE", JobManager.SLOTS_AVAILABLE)

    scheduler = PostgreSQLJobScheduler(config=scheduler_config, job_manager=job_manager)
    await scheduler.start()
    assert scheduler.running
    await scheduler.wipe()

    return scheduler


@pytest.mark.asyncio
async def test_list_jobs_empty(job_scheduler: JobScheduler):
    assert await job_scheduler.list_jobs() == dict()


@pytest.mark.asyncio
async def test_list_work_items(job_scheduler: JobScheduler):
    items = await job_scheduler.list_jobs()
    assert items == dict()

    w1 = build_work_item("queue-001", "1")
    w2 = build_work_item("queue-001", "2")

    await job_scheduler.submit_job(w1)
    await job_scheduler.submit_job(w2)

    items = await job_scheduler.list_jobs()
    assert len(items.items()) == 2

    r1 = await job_scheduler.get_job(w1.id)
    r2 = await job_scheduler.get_job(w2.id)

    assert 0 == len(compare_pydantic_models(w1, r1, ["keep_until", "start_after"]))
    assert 0 == len(compare_pydantic_models(w2, r2, ["keep_until", "start_after"]))

    _ = asyncio.create_task(
        async_delay(
            update_job_scheduler_status(job_scheduler, "1", WorkState.COMPLETED), 1
        )
    )

    _ = asyncio.create_task(
        async_delay(
            update_job_scheduler_status(job_scheduler, "2", WorkState.COMPLETED), 1
        )
    )

    await async_wait_for_condition_async_predicate(
        check_job_scheduler_succeeded, job_scheduler=job_scheduler, job_id="1"
    )
    await async_wait_for_condition_async_predicate(
        check_job_scheduler_succeeded, job_scheduler=job_scheduler, job_id="2"
    )

    jobs_info = await job_scheduler.list_jobs()
    assert "1" in jobs_info
    assert jobs_info["1"]._worker_state == WorkState.COMPLETED

    assert "2" in jobs_info
    assert jobs_info["2"]._worker_state == WorkState.COMPLETED


@pytest.mark.asyncio
async def test_pass_job_id(job_scheduler: JobScheduler):
    submission_id = "my_custom_id"

    returned_id = await job_scheduler.submit_job(
        build_work_item("queue-001", submission_id)
    )
    assert returned_id == submission_id

    _ = asyncio.create_task(
        async_delay(
            update_job_scheduler_status(
                job_scheduler, submission_id, WorkState.COMPLETED
            ),
            1,
        )
    )

    await async_wait_for_condition_async_predicate(
        check_job_scheduler_succeeded, job_scheduler=job_scheduler, job_id=submission_id
    )

    # Check that the same job_id is rejected.
    with pytest.raises(ValueError):
        await job_scheduler.submit_job(build_work_item("queue-001", submission_id))


@pytest.mark.asyncio
async def test_simultaneous_submit_job(job_scheduler: JobScheduler):
    """Test that we can submit multiple jobs at once."""
    job_ids = await asyncio.gather(
        job_scheduler.submit_job(build_work_item("queue-001")),
        job_scheduler.submit_job(build_work_item("queue-001")),
        job_scheduler.submit_job(build_work_item("queue-001")),
    )

    for job_id in job_ids:
        _ = asyncio.create_task(
            async_delay(
                update_job_scheduler_status(job_scheduler, job_id, WorkState.COMPLETED),
                1,
            )
        )

        await async_wait_for_condition_async_predicate(
            check_job_scheduler_succeeded, job_scheduler=job_scheduler, job_id=job_id
        )


@pytest.mark.asyncio
async def test_simultaneous_with_same_id(job_scheduler: JobScheduler):
    """Test that we can submit multiple jobs at once with the same id.

    The second job should raise a friendly error.
    """
    with pytest.raises(ValueError) as excinfo:
        await asyncio.gather(
            job_scheduler.submit_job(build_work_item("queue-001", "1")),
            job_scheduler.submit_job(build_work_item("queue-001", "1")),
        )
    assert "Job with submission_id 1 already exists" in str(excinfo.value)

    # Check that the (first) job can still succeed.
    _ = asyncio.create_task(
        async_delay(
            update_job_scheduler_status(job_scheduler, "1", WorkState.COMPLETED), 1
        )
    )

    await async_wait_for_condition_async_predicate(
        check_job_scheduler_succeeded, job_scheduler=job_scheduler, job_id="1"
    )


@pytest.mark.asyncio
async def test_job_scheduler_submission(
        job_scheduler: JobScheduler, job_manager: JobManager
):
    JobManager.SLOTS_AVAILABLE = 1

    job_id = await job_scheduler.submit_job(build_work_item("queue-001", "1"))
    assert job_id is not None

    job_info = await job_manager.get_job_info(job_id)
    assert job_info is not None

    work_info = await job_scheduler.get_job(job_id)
    assert work_info is not None

    assert work_info.id == job_id
    assert job_info.status == JobStatus.RUNNING
    assert work_info._worker_state == WorkState.CREATED


@pytest.mark.asyncio
async def test_job_scheduler_completion(
        job_scheduler: JobScheduler, job_manager: JobManager
):
    JobManager.SLOTS_AVAILABLE = 1

    async def handle_event(message: Any):
        print(f"received message: {message}")

    job_manager.event_publisher.subscribe(
        [JobStatus.SUCCEEDED, JobStatus.FAILED], handle_event
    )

    job_id = await job_scheduler.submit_job(build_work_item("queue-001", "1"))
    assert job_id is not None

    job_info = await job_manager.get_job_info(job_id)
    assert job_info is not None

    work_info = await job_scheduler.get_job(job_id)
    assert work_info is not None

    assert work_info.id == job_id
    assert job_info.status == JobStatus.RUNNING
    assert work_info._worker_state == WorkState.CREATED
