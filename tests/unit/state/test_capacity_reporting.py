from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from marie.state.semaphore_store import CapacityDoc, SemaphoreHolder, SemaphoreStore
from marie.state.slot_capacity_manager import SlotCapacityManager


def reporting_store(values: dict[str, bytes]) -> SemaphoreStore:
    # A read-only backend. Any unexpected write during reporting fails the test.
    etcd = Mock(spec_set=["client", "get", "_mangle_key", "_demangle_key"])
    etcd.client = Mock(spec_set=["get_prefix"])
    etcd._mangle_key.side_effect = str.encode
    etcd._demangle_key.side_effect = bytes.decode
    etcd.get.side_effect = lambda key, **kwargs: (values.get(key), None)
    etcd.client.get_prefix.side_effect = lambda prefix: [
        (raw, SimpleNamespace(key=key.encode()))
        for key, raw in values.items()
        if key.startswith(prefix.decode())
    ]
    return SemaphoreStore(etcd)


def test_refresh_collects_one_report_after_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARIE_CAPACITY_ZERO_ON_ABSENT", "true")
    monkeypatch.setenv("MARIE_SLOTS_LLM_PER_NODE", "1")
    store = reporting_store({"capacity/llm": b'{"limit":1,"requested_limit":1}'})
    manager = SlotCapacityManager(store)

    rows, totals, summary = manager.refresh_from_nodes(
        {"llm": [{"address": "worker:1"}]}
    )

    assert (store.etcd.client.get_prefix.call_count, store.etcd.get.call_count) == (
        4,
        2,
    )
    assert rows == [("llm", 1, 1, 0, 1, 0, "")]
    assert totals == {"capacity": 1, "used": 0, "available": 1, "holder_count": 0}
    table = [
        [cell.strip() for cell in line.split("|")] for line in summary.splitlines()
    ]
    assert table[3] == ["llm", "1", "1", "0", "1", "0", ""]
    assert table[-1] == ["TOTAL", "1", "", "0", "1", "0", ""]

    store.etcd.reset_mock()
    assert manager._format_summary(rows, totals) == summary
    assert not store.etcd.mock_calls


def test_snapshot_keeps_unknown_keys_and_malformed_records() -> None:
    store = reporting_store(
        {
            "capacity/llm": b"null",
            "semaphores/llm/count": b"invalid",
            "semaphores/llm/holders/ticket": b"invalid",
            "semaphores/old/legacy": b"unused",
            "semaphores/ignored": b"outside slot inventory",
        }
    )
    snapshot = store.snapshot_all(include_holders=True)
    assert set(snapshot) == {"llm", "old"}
    assert snapshot["old"] == {
        "capacity": 0,
        "target": 0,
        "used": 0,
        "available": 0,
        "holder_count": 0,
        "holders": {},
    }
    assert snapshot["llm"] == {**snapshot["old"], "holder_count": 1}
    assert store.etcd.client.get_prefix.call_count == 2

    store.etcd.reset_mock()
    assert store.available_count_all() == {"llm": 0}
    assert store.etcd.client.get_prefix.call_count == 2


def test_public_formatter_reads_current_inventory() -> None:
    values = {}
    manager = SlotCapacityManager(reporting_store(values))
    rows, totals, summary = manager.refresh_from_nodes({})
    assert rows == []
    assert totals == {"capacity": 0, "used": 0, "available": 0, "holder_count": 0}
    assert summary == "[capacity] No slots discovered."
    assert manager.format_summary() == summary

    values["capacity/llm"] = b'{"limit":2,"requested_limit":1}'
    cells = manager.format_summary().splitlines()[3].split("|")
    assert [cell.strip() for cell in cells] == ["llm", "2", "1", "0", "1", "0", ""]
    cells = manager.format_summary(targets={"llm": 0}).splitlines()[3].split("|")
    assert [cell.strip() for cell in cells] == ["llm", "2", "0", "0", "1", "0", ""]


def test_aggregate_helpers_preserve_their_slot_inventories() -> None:
    store = reporting_store(
        {
            "capacity/idle": b'{"limit":3}',
            "capacity/draining": b'{"limit":5,"requested_limit":1}',
            "capacity/bad": b"null",
            "semaphores/draining/count": b"2",
            "semaphores/count-only/count": b" -3 ",
            "semaphores/bad/count": b"4\x1c",
            "semaphores/holder-only/holders/t": (
                b'{"job_id":"t","node":"n","ttl":12,"created_at":"then"}'
            ),
            "semaphores/holder-only/holders/bad": b"null",
            "semaphores/unknown/legacy/field": b"unused",
        }
    )
    assert store.capacities_all() == {"idle": 3, "draining": 5, "bad": 0}
    assert store.available_count_all() == {
        "idle": 3,
        "draining": 0,
        "bad": 0,
        "count-only": 0,
    }
    assert store.read_count_all() == {
        "idle": 0,
        "draining": 2,
        "bad": 0,
        "count-only": -3,
        "holder-only": 0,
        "unknown": 0,
    }
    assert store.holder_counts_all() == {
        "idle": 0,
        "draining": 0,
        "bad": 0,
        "count-only": 0,
        "holder-only": 2,
        "unknown": 0,
    }
    assert store.list_holders_all() == {
        "idle": {},
        "draining": {},
        "bad": {},
        "count-only": {},
        "unknown": {},
        "holder-only": {
            "t": SemaphoreHolder(ticket_id="t", node="n", ttl=12, created_at="then")
        },
    }
    snapshot = store.snapshot_all()
    assert snapshot["count-only"]["used"] == 0
    assert snapshot["bad"]["used"] == 0


def test_capacity_and_availability_scan_only_the_needed_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = reporting_store(
        {
            "capacity/llm": b'{"limit":3}',
            "semaphores/llm/count": b"1",
            "semaphores/llm/holders/t": b"{}",
        }
    )
    decode_holder = Mock(wraps=SemaphoreHolder.from_json)
    monkeypatch.setattr(SemaphoreHolder, "from_json", decode_holder)
    assert store.capacities_all() == {"llm": 3}
    assert store.etcd.client.get_prefix.call_args_list == [call(b"capacity/")]
    store.etcd.reset_mock()
    assert store.available_count_all() == {"llm": 2}
    assert store.etcd.client.get_prefix.call_args_list == [
        call(b"capacity/"),
        call(b"semaphores/"),
    ]
    decode_holder.assert_not_called()


def test_inventory_reports_scan_each_prefix_once_without_decoding_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = reporting_store({"capacity/llm": b"invalid"})
    decode_capacity = Mock(wraps=CapacityDoc.from_json)
    monkeypatch.setattr(CapacityDoc, "from_json", decode_capacity)
    for report, expected in (
        (store.read_count_all, {"llm": 0}),
        (store.holder_counts_all, {"llm": 0}),
        (store.list_holders_all, {"llm": {}}),
    ):
        store.etcd.reset_mock()
        assert report() == expected
        assert store.etcd.client.get_prefix.call_args_list == [
            call(b"capacity/"),
            call(b"semaphores/"),
        ]
    decode_capacity.assert_not_called()
