import asyncio

import psycopg
import pytest

import marie.scheduler.services.notification_service as notification_service_module
from marie.scheduler.services.notification_service import NotificationService


class FakeConnection:
    def __init__(self):
        self.closed = False
        self.executed = []

    async def execute(self, query, params=None):
        statement = query.as_string() if hasattr(query, "as_string") else query
        self.executed.append((statement, params))

    async def close(self):
        self.closed = True

    async def notifies(self, **_kwargs):
        if False:
            yield None


@pytest.fixture
def config():
    return {
        "username": "user",
        "password": "pass",
        "database": "db",
        "hostname": "localhost",
        "port": 5432,
        "application_name": "scheduler-test",
    }


@pytest.mark.asyncio
async def test_setup_connection_uses_keepalive_and_registers_channels(
    monkeypatch, config
):
    service = NotificationService(config)
    service.register_handler("dag_state_changed", lambda _payload: None)

    captured = {}
    connection = FakeConnection()

    async def fake_connect(**kwargs):
        captured.update(kwargs)
        return connection

    monkeypatch.setattr(
        notification_service_module.psycopg.AsyncConnection,
        "connect",
        fake_connect,
    )

    await service._setup_connection()

    assert captured["keepalives"] == 1
    assert captured["keepalives_idle"] == 60
    assert captured["keepalives_interval"] == 10
    assert captured["keepalives_count"] == 5
    assert captured["application_name"] == "scheduler-test_listener"
    assert captured["autocommit"] is True
    assert connection.executed == [('LISTEN "dag_state_changed"', None)]


@pytest.mark.asyncio
async def test_notification_listener_reconnects_after_runtime_failure(
    monkeypatch, config
):
    service = NotificationService(config)
    service.register_handler("dag_state_changed", lambda _payload: None)
    service.running = True

    setup_calls = []
    close_calls = []
    sleep_calls = []
    connections = [FakeConnection(), FakeConnection()]

    async def fake_setup():
        connection = connections[len(setup_calls)]
        service._listen_connection = connection
        setup_calls.append(connection)

    async def fake_close():
        close_calls.append(True)
        if service._listen_connection is not None:
            service._listen_connection.closed = True
        service._listen_connection = None

    notification_calls = {"count": 0}

    async def fake_next_notification():
        notification_calls["count"] += 1
        if notification_calls["count"] == 1:
            raise RuntimeError("socket gone")
        service.running = False
        return None

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(service, "_setup_connection", fake_setup)
    monkeypatch.setattr(service, "_close_connection", fake_close)
    monkeypatch.setattr(service, "_next_notification", fake_next_notification)
    monkeypatch.setattr(notification_service_module.asyncio, "sleep", fake_sleep)

    await service._listen_for_notifications()

    assert len(setup_calls) == 2
    assert len(close_calls) >= 2
    assert sleep_calls == [service._reconnect_base_delay]
    assert service.connected is False
    assert service._ever_connected is True


@pytest.mark.asyncio
async def test_next_notification_preserves_all_notifications_from_receive_batch(config):
    notifications = [
        psycopg.Notify("job_terminal", '{"job_id":"1"}', 1),
        psycopg.Notify("job_terminal", '{"job_id":"2"}', 1),
        psycopg.Notify("job_terminal", '{"job_id":"3"}', 1),
    ]

    class BurstConnection(FakeConnection):
        def __init__(self):
            super().__init__()
            self.notifies_calls = 0

        async def notifies(self, **_kwargs):
            self.notifies_calls += 1
            for notification in notifications:
                yield notification

    service = NotificationService(config)
    connection = BurstConnection()
    service._listen_connection = connection

    received = [await service._next_notification() for _ in notifications]

    assert received == notifications
    assert connection.notifies_calls == 1


@pytest.mark.asyncio
async def test_listener_traces_driver_and_handler_time(monkeypatch, config) -> None:
    notification = psycopg.Notify(
        "job_terminal",
        '{"job_id":"job-1","status":"SUCCEEDED"}',
        1,
    )

    class SingleNotificationConnection(FakeConnection):
        async def notifies(self, **_kwargs):
            yield notification

    service = NotificationService(config)

    async def handler(_payload):
        service.running = False

    service.register_handler("job_terminal", handler)
    service.running = True
    connection = SingleNotificationConnection()
    events: list[tuple[str, dict]] = []

    async def fake_setup():
        service._listen_connection = connection

    async def fake_close():
        service._listen_connection = None

    monkeypatch.setattr(service, "_setup_connection", fake_setup)
    monkeypatch.setattr(service, "_close_connection", fake_close)
    monkeypatch.setattr(
        notification_service_module,
        "scheduler_trace",
        lambda event, **fields: events.append((event, fields)),
    )

    await service._listen_for_notifications()

    assert [event for event, _fields in events] == [
        "postgres_notification_handler_completed"
    ]
    fields = events[0][1]
    assert fields["job_id"] == "job-1"
    assert fields["succeeded"] is True
    assert fields["driver_to_dispatch_ms"] >= 0
    assert fields["handler_ms"] >= 0


@pytest.mark.asyncio
async def test_send_notification_uses_async_connection_and_closes_it(
    monkeypatch, config
):
    service = NotificationService(config)
    connection = FakeConnection()

    async def fake_connect(**_kwargs):
        return connection

    monkeypatch.setattr(
        notification_service_module.psycopg.AsyncConnection,
        "connect",
        fake_connect,
    )

    sent = await service.send_notification("dag_state_changed", {"dag_id": "1"})

    assert sent is True
    assert connection.executed == [
        (
            "SELECT pg_notify(%s, %s)",
            ("dag_state_changed", '{"dag_id": "1"}'),
        )
    ]
    assert connection.closed is True
