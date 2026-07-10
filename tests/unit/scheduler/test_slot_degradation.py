from unittest.mock import Mock

from marie.scheduler.util import available_slots_by_executor
from marie.state.semaphore_store import EtcdStoreUnavailable, SemaphoreStore


def test_degrades_to_empty_snapshot_when_store_unavailable(mocker):
    sem = mocker.Mock(spec=SemaphoreStore)
    sem.available_count_all.side_effect = EtcdStoreUnavailable("channel closed")

    assert available_slots_by_executor(sem) == {}


def test_passthrough_when_store_healthy(mocker):
    sem = mocker.Mock(spec=SemaphoreStore)
    sem.available_count_all.return_value = {"extract_executor": 2}

    assert available_slots_by_executor(sem) == {"extract_executor": 2}


def test_other_errors_still_propagate(mocker):
    import pytest

    sem = mocker.Mock(spec=SemaphoreStore)
    sem.available_count_all.side_effect = KeyError("bug")

    with pytest.raises(KeyError):
        available_slots_by_executor(sem)
