import threading
from types import SimpleNamespace
from unittest.mock import Mock

from marie.serve.discovery.resolver import EtcdServiceResolver


def test_watch_service_signals_snapshot_after_initial_events():
    resolver = object.__new__(EtcdServiceResolver)
    resolver._lock = threading.Lock()
    resolver._watch_states = {}
    resolver._names = {}
    resolver._add_watch_with_options = Mock(return_value=1)
    resolver.get = Mock(return_value=[])
    order = []

    def fire_initial_events(service_name, callback):
        callback(
            service_name,
            SimpleNamespace(values=None, event="put", key="node-1"),
        )
        callback(
            service_name,
            SimpleNamespace(values=None, event="put", key="node-2"),
        )
        return 2

    resolver._fire_initial_events = fire_initial_events

    resolver.watch_service(
        "gateway/marie",
        lambda _service, event: order.append(event.key),
        initial_snapshot_callback=lambda _service, count: order.append(
            f"snapshot:{count}"
        ),
    )

    assert order == ["node-1", "node-2", "snapshot:2"]


def test_watch_service_signals_empty_snapshot():
    resolver = object.__new__(EtcdServiceResolver)
    resolver._lock = threading.Lock()
    resolver._watch_states = {}
    resolver._names = {}
    resolver._add_watch_with_options = Mock(return_value=1)
    resolver.get = Mock(return_value=[])
    resolver._fire_initial_events = Mock(return_value=0)
    complete = Mock()

    resolver.watch_service(
        "gateway/marie",
        Mock(),
        initial_snapshot_callback=complete,
    )

    complete.assert_called_once_with("gateway/marie", 0)
