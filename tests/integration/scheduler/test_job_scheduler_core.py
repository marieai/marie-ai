from datetime import datetime

import pytest

from marie_server.scheduler import PostgreSQLJobScheduler
from marie_server.scheduler.jobscheduler import JobScheduler
from marie_server.scheduler.models import WorkInfo


@pytest.fixture
def num_jobs(request):
    print("request.param", request.param)
    return request.param


def build_work_item(name: str) -> WorkInfo:
    return WorkInfo(
        name=name,
        priority=0,
        data={},
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
# @pytest.mark.asyncio
# async def test_list_jobs_empty(job_manager: JobManager):
#     assert await job_manager.list_jobs() == dict()


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
    return scheduler


@pytest.mark.asyncio
async def test_list_work_items(job_scheduler: JobScheduler):
    items = await job_scheduler.list_jobs()
    assert items == []

    w1 = await job_scheduler.put_job(build_work_item("WorkInfo-001"))
    w2 = await job_scheduler.put_job(build_work_item("WorkInfo-002"))

    assert w1 == job_scheduler.get_job(w1)
    assert w2 == job_scheduler.get_job(w2)

    items = await job_scheduler.list_jobs()
    assert len(items) == 2


@pytest.mark.asyncio
# @pytest.mark.parametrize("num_jobs", [1], indirect=True)
async def test_job_scheduler_setup(job_scheduler: PostgreSQLJobScheduler):
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

    work_info = WorkInfo(
        name="WorkInfo-001",
        priority=0,
        data={},
        retryLimit=0,
        retryDelay=0,
        retryBackoff=False,
        startAfter=datetime.now(),
        expireInSeconds=0,
        singletonKey="",
        keepUntil=datetime.now(),
        onComplete=False,
    )
    #
    # job_id = scheduler.schedule(work_info)
    # print("job_id", job_id)
