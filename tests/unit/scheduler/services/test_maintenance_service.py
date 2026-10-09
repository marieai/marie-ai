from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from marie.scheduler.services.maintenance_service import MaintenanceService


@pytest.mark.parametrize('operation', ['archive', 'purge'])
async def test_unsupported_retention_operations_raise(operation: str) -> None:
    service = MaintenanceService(repository=SimpleNamespace())

    with pytest.raises(NotImplementedError):
        await getattr(service, operation)()


async def test_periodic_maintenance_recovers_leases_without_retention_errors() -> None:
    repository = SimpleNamespace(
        release_expired_leases=AsyncMock(return_value=1),
        recover_expired_run_leases=AsyncMock(return_value=[]),
    )
    notified = False

    async def notify() -> bool:
        nonlocal notified
        notified = True
        return True

    service = MaintenanceService(repository=repository, notify_callback=notify)
    service.logger = Mock()

    await service.maintenance()

    assert notified
    service.logger.error.assert_not_called()
