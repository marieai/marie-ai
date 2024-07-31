from datetime import datetime
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from marie_server.scheduler import PostgreSQLJobScheduler
from marie_server.scheduler.job_scheduler import JobScheduler
from marie_server.scheduler.models import WorkInfo
from marie_server.scheduler.state import States


def compare_pydantic_models(model1: BaseModel, model2: BaseModel, excludes: List) -> Dict[str, Any]:
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
            differences[key] = {'model1': dict1.get(key), 'model2': dict2.get(key)}

    return differences


@pytest.fixture
def num_jobs(request):
    print("request.param", request.param)
    return request.param


def build_work_item(name: str, job_id: str) -> WorkInfo:
    return WorkInfo(
        id=job_id,
        name=name,
        priority=0,
        data={},
        state=States.CREATED,
        retry_limit=0,
        retry_delay=0,
        retry_backoff=False,
        start_after=datetime.now(),
        expire_in_seconds=0,
        keep_until=datetime.now(),
        on_complete=False,
    )


# @pytest.mark.asyncio
# @pytest.fixture
# async def job_manager(tmp_path):
#     storage = InMemoryKV()
#     # TODO: Externalize the storage configuration
#     storage_config = {
#         "hostname": "127.0.0.1",
#         "port": 5432,
#         "username": "postgres",
#         "password": "123456",
#         "database": "postgres",
#         "default_table": "kv_store_a",
#         "max_pool_size": 5,
#         "max_connections": 5,
#     }
#
#     storage = PostgreSQLKV(config=storage_config, reset=True)
#     yield JobManager(storage=storage, job_distributor=NoopJobDistributor())

#
#
@pytest.mark.asyncio
async def test_list_jobs_empty(job_scheduler: JobScheduler):
    assert await job_scheduler.list_jobs() == dict()


@pytest.mark.asyncio
@pytest.fixture
async def job_scheduler(tmp_path):
    scheduler_config = {
        "hostname": "localhost",
        "port": 5432,
        "database": "postgres",
        "username": "postgres",
        "password": "123456",
    }

    scheduler = PostgreSQLJobScheduler(config=scheduler_config, job_manager=None)
    await scheduler.start()
    assert scheduler.running
    await scheduler.wipe()

    return scheduler


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

    print("r1", r1)
    print("r2", r2)


@pytest.mark.asyncio
# @pytest.mark.parametrize("num_jobs", [1], indirect=True)
async def test_job_scheduler_setup(job_scheduler: JobScheduler):
    work_info = WorkInfo(
        name="WorkInfo-001",
        priority=0,
        data={},
        state=States.CREATED,
        retry_limit=0,
        retry_delay=0,
        retry_backoff=False,
        start_after=datetime.now(),
        expire_in_seconds=0,
        keep_until=datetime.now(),
        on_complete=False,
    )
    #
    # job_id = scheduler.schedule(work_info)
    # print("job_id", job_id)
