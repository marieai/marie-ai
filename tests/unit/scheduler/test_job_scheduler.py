from typing import Any

import pytest

from marie.scheduler.job_scheduler import JobScheduler


def test_scheduler_without_reset_implementation_cannot_be_instantiated() -> None:
    def unsupported_operation(self: JobScheduler, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    implementations = {
        name: unsupported_operation
        for name in JobScheduler.__abstractmethods__
        if name != 'reset_active_dags'
    }
    scheduler_type = type('MissingResetScheduler', (JobScheduler,), implementations)

    with pytest.raises(TypeError, match='reset_active_dags'):
        scheduler_type()
