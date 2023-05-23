import asyncio
import sys

import pytest

from marie_server.job.common import JobStatus
from marie_server.job.job_manager import generate_job_id, JobManager
from marie_server.storage.in_memory import InMemoryKV
from tests.core.test_utils import async_wait_for_condition_async_predicate


async def check_job_succeeded(job_manager, job_id):
    data = await job_manager.get_job_info(job_id)
    status = data.status
    if status == JobStatus.FAILED:
        raise RuntimeError(f"Job failed! {data.message}")
    assert status in {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED}
    return status == JobStatus.SUCCEEDED


async def check_job_failed(job_manager, job_id):
    status = await job_manager.get_job_status(job_id)
    assert status in {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.FAILED}
    return status == JobStatus.FAILED


async def check_job_stopped(job_manager, job_id):
    status = await job_manager.get_job_status(job_id)
    assert status in {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.STOPPED}
    return status == JobStatus.STOPPED


async def check_job_running(job_manager, job_id):
    status = await job_manager.get_job_status(job_id)
    assert status in {JobStatus.PENDING, JobStatus.RUNNING}
    return status == JobStatus.RUNNING


@pytest.mark.asyncio
@pytest.fixture
async def job_manager(tmp_path):
    storage = InMemoryKV()
    yield JobManager(storage)


def test_generate_job_id():
    ids = set()
    for _ in range(10000):
        new_id = generate_job_id()
        assert "-" not in new_id
        ids.add(new_id)

    assert len(ids) == 10000


@pytest.mark.asyncio
async def test_list_jobs_empty(job_manager: JobManager):
    assert await job_manager.list_jobs() == dict()


@pytest.mark.asyncio
async def test_list_jobs(job_manager: JobManager):
    await job_manager.submit_job(entrypoint="echo hi", submission_id="1")

    runtime_env = {"env_vars": {"TEST": "123"}}
    metadata = {"foo": "bar"}
    await job_manager.submit_job(
        entrypoint="echo hello",
        submission_id="2",
        runtime_env=runtime_env,
        metadata=metadata,
    )
    await async_wait_for_condition_async_predicate(
        check_job_succeeded, job_manager=job_manager, job_id="1"
    )
    await async_wait_for_condition_async_predicate(
        check_job_succeeded, job_manager=job_manager, job_id="2"
    )
    jobs_info = await job_manager.list_jobs()
    assert "1" in jobs_info
    assert jobs_info["1"].status == JobStatus.SUCCEEDED

    assert "2" in jobs_info
    assert jobs_info["2"].status == JobStatus.SUCCEEDED
    assert jobs_info["2"].message is not None
    assert jobs_info["2"].end_time >= jobs_info["2"].start_time
    assert jobs_info["2"].runtime_env == runtime_env
    assert jobs_info["2"].metadata == metadata


@pytest.mark.asyncio
async def test_pass_job_id(job_manager):
    submission_id = "my_custom_id"

    returned_id = await job_manager.submit_job(
        entrypoint="echo hello", submission_id=submission_id
    )
    assert returned_id == submission_id

    await async_wait_for_condition_async_predicate(
        check_job_succeeded, job_manager=job_manager, job_id=submission_id
    )

    # Check that the same job_id is rejected.
    with pytest.raises(ValueError):
        await job_manager.submit_job(
            entrypoint="echo hello", submission_id=submission_id
        )


@pytest.mark.asyncio
async def test_simultaneous_submit_job(job_manager):
    """Test that we can submit multiple jobs at once."""
    job_ids = await asyncio.gather(
        job_manager.submit_job(entrypoint="echo hello"),
        job_manager.submit_job(entrypoint="echo hello"),
        job_manager.submit_job(entrypoint="echo hello"),
    )

    for job_id in job_ids:
        await async_wait_for_condition_async_predicate(
            check_job_succeeded, job_manager=job_manager, job_id=job_id
        )


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))

# pip install pytest-asyncio
