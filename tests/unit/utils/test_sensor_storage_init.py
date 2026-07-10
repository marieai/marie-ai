import asyncio
import threading

import pytest

from marie.utils import server_runtime


class _FakePool:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _FakeSensorWorker:
    def __init__(self):
        self.storage = None
        self._daemon_task = None

    def set_storage(self, storage):
        self.storage = storage

    async def start(self):
        return None


@pytest.fixture(autouse=True)
def _reset_globals():
    """Ensure module globals don't leak between tests."""
    server_runtime._sensor_worker = None
    server_runtime._sensor_storage_pool = None
    server_runtime._sensor_worker_loop = None
    yield
    server_runtime._sensor_worker = None
    server_runtime._sensor_storage_pool = None
    server_runtime._sensor_worker_loop = None


def test_init_sensor_storage_wires_pool_and_schema(monkeypatch):
    fake_worker = _FakeSensorWorker()
    server_runtime._sensor_worker = fake_worker

    fake_pool = _FakePool()
    create_pool_calls = []

    async def fake_create_pool(**kwargs):
        create_pool_calls.append(kwargs)
        return fake_pool

    initialize_calls = []

    class _FakeStorage:
        pass

    def fake_initialize(pool, schema):
        initialize_calls.append((pool, schema))
        return _FakeStorage()

    monkeypatch.setattr("asyncpg.create_pool", fake_create_pool)
    monkeypatch.setattr(
        "marie.sensors.state.psql_storage.PostgreSQLSensorStorage.initialize",
        staticmethod(fake_initialize),
    )

    db_config = {
        "hostname": "db.internal",
        "port": "5432",
        "username": "marie",
        "password": "secret",
        "database": "marie_scheduler_db",
        "schema": "marie_scheduler",
        "min_connections": 2,
        "max_connections": 7,
    }

    asyncio.run(server_runtime._init_sensor_storage(db_config))

    assert len(create_pool_calls) == 1
    call = create_pool_calls[0]
    assert call["host"] == "db.internal"
    assert call["port"] == 5432
    assert call["user"] == "marie"
    assert call["password"] == "secret"
    assert call["database"] == "marie_scheduler_db"
    assert call["min_size"] == 2
    assert call["max_size"] == 7

    assert initialize_calls == [(fake_pool, "marie_scheduler")]
    assert isinstance(fake_worker.storage, _FakeStorage)
    assert server_runtime._sensor_storage_pool is fake_pool


def test_init_sensor_storage_defaults_schema(monkeypatch):
    fake_worker = _FakeSensorWorker()
    server_runtime._sensor_worker = fake_worker

    fake_pool = _FakePool()
    initialize_calls = []

    async def fake_create_pool(**kwargs):
        return fake_pool

    def fake_initialize(pool, schema):
        initialize_calls.append((pool, schema))
        return object()

    monkeypatch.setattr("asyncpg.create_pool", fake_create_pool)
    monkeypatch.setattr(
        "marie.sensors.state.psql_storage.PostgreSQLSensorStorage.initialize",
        staticmethod(fake_initialize),
    )

    db_config = {
        "hostname": "db.internal",
        "port": 5432,
        "username": "marie",
        "password": "secret",
        "database": "marie_scheduler_db",
    }

    asyncio.run(server_runtime._init_sensor_storage(db_config))

    assert initialize_calls == [(fake_pool, "marie_scheduler")]


def test_init_sensor_storage_noop_when_db_config_empty():
    fake_worker = _FakeSensorWorker()
    server_runtime._sensor_worker = fake_worker

    asyncio.run(server_runtime._init_sensor_storage(None))
    asyncio.run(server_runtime._init_sensor_storage({}))

    assert fake_worker.storage is None
    assert server_runtime._sensor_storage_pool is None


def test_init_sensor_storage_noop_when_worker_none(monkeypatch):
    server_runtime._sensor_worker = None

    async def fake_create_pool(**kwargs):
        raise AssertionError("create_pool should not be called when worker is None")

    monkeypatch.setattr("asyncpg.create_pool", fake_create_pool)

    asyncio.run(server_runtime._init_sensor_storage({"hostname": "db.internal"}))

    assert server_runtime._sensor_storage_pool is None


def test_init_sensor_storage_logs_and_swallows_errors(monkeypatch, caplog):
    fake_worker = _FakeSensorWorker()
    server_runtime._sensor_worker = fake_worker

    async def failing_create_pool(**kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("asyncpg.create_pool", failing_create_pool)

    # Should not raise.
    asyncio.run(server_runtime._init_sensor_storage({"hostname": "db.internal"}))

    assert fake_worker.storage is None
    assert server_runtime._sensor_storage_pool is None


def test_run_sensor_worker_closes_pool_on_shutdown(monkeypatch):
    fake_worker = _FakeSensorWorker()
    server_runtime._sensor_worker = fake_worker

    fake_pool = _FakePool()

    async def fake_init_sensor_storage(db_config):
        server_runtime._sensor_storage_pool = fake_pool

    monkeypatch.setattr(server_runtime, "_init_sensor_storage", fake_init_sensor_storage)

    server_runtime._run_sensor_worker({"hostname": "db.internal"})

    assert fake_pool.closed is True
    assert server_runtime._sensor_storage_pool is None


class _FakeSensorWorkerForAttach:
    def __init__(self):
        self.job_scheduler = None

    def set_job_scheduler(self, job_scheduler):
        self.job_scheduler = job_scheduler


class _FakeGatewayScheduler:
    """Stands in for PostgreSQLJobScheduler: records which loop ran it."""

    def __init__(self):
        self.executed_on_loop = None

    async def submit_job(self, work_info):
        self.executed_on_loop = asyncio.get_running_loop()
        await asyncio.sleep(0)
        return f"job-for-{work_info}"


def test_attach_sensor_worker_scheduler_noop_when_no_worker():
    server_runtime._sensor_worker = None

    assert server_runtime.attach_sensor_worker_scheduler(object()) is False


def test_attach_sensor_worker_scheduler_sets_worker_job_scheduler():
    fake_worker = _FakeSensorWorkerForAttach()
    server_runtime._sensor_worker = fake_worker
    fake_scheduler = _FakeGatewayScheduler()

    async def _call_attach():
        return server_runtime.attach_sensor_worker_scheduler(fake_scheduler)

    attached = asyncio.run(_call_attach())

    assert attached is True
    assert isinstance(fake_worker.job_scheduler, server_runtime._CrossLoopJobScheduler)
    assert fake_worker.job_scheduler._job_scheduler is fake_scheduler


def test_attach_sensor_worker_scheduler_bridges_two_real_loops():
    """
    The adapter's submit_job runs on the WORKER's loop (as worker.py awaits
    it there), hands the real coroutine to the GATEWAY's loop via
    run_coroutine_threadsafe, and must propagate the result back to the
    worker loop caller. Exercise this with two real event loops, each
    running in its own thread, matching production topology.
    """
    fake_worker = _FakeSensorWorkerForAttach()
    server_runtime._sensor_worker = fake_worker
    fake_scheduler = _FakeGatewayScheduler()

    gateway_loop = asyncio.new_event_loop()
    worker_loop = asyncio.new_event_loop()
    server_runtime._sensor_worker_loop = worker_loop

    gateway_thread = threading.Thread(target=gateway_loop.run_forever, daemon=True)
    worker_thread = threading.Thread(target=worker_loop.run_forever, daemon=True)
    gateway_thread.start()
    worker_thread.start()

    try:
        async def _call_attach():
            return server_runtime.attach_sensor_worker_scheduler(fake_scheduler)

        # attach() must observe the gateway loop as "running", so it is
        # itself invoked as a coroutine on that loop.
        attached = asyncio.run_coroutine_threadsafe(
            _call_attach(), gateway_loop
        ).result(timeout=5)
        assert attached is True

        adapter = fake_worker.job_scheduler
        assert adapter is not None

        # Submit "from" the worker's loop, the way SensorWorker._submit_run_request does.
        result = asyncio.run_coroutine_threadsafe(
            adapter.submit_job("work-1"), worker_loop
        ).result(timeout=5)

        assert result == "job-for-work-1"
        assert fake_scheduler.executed_on_loop is gateway_loop
    finally:
        gateway_loop.call_soon_threadsafe(gateway_loop.stop)
        worker_loop.call_soon_threadsafe(worker_loop.stop)
        gateway_thread.join(timeout=5)
        worker_thread.join(timeout=5)
        gateway_loop.close()
        worker_loop.close()
