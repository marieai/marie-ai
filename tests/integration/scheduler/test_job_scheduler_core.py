import pytest

from marie_server.scheduler import PostgreSQLJobScheduler


@pytest.fixture
def num_jobs(request):
    print("request.param", request.param)
    return request.param


@pytest.mark.parametrize("num_jobs", [1], indirect=True)
def test_job_scheduler_setup(num_jobs):
    print("num_jobs", num_jobs)
    scheduler_config = {
        "hostname": "localhost",
        "port": 5432,
        "database": "postgres",
        "username": "postgres",
        "password": "123456",
    }

    scheduler = PostgreSQLJobScheduler(config=scheduler_config)
    scheduler.start_schedule()
    print(scheduler)

    assert scheduler.running
