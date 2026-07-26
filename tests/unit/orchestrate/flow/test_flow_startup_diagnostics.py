import asyncio
from collections.abc import Iterator
from unittest.mock import Mock

import pytest

from marie.excepts import RuntimeFailToStart
from marie.orchestrate.flow.base import Flow


class _FailingDeployment:
    external = False

    async def async_wait_start_success(self) -> None:
        raise OSError(98, "Address already in use")


class _PendingDeployment:
    external = False

    async def async_wait_start_success(self) -> None:
        await asyncio.Event().wait()


class _FlowWithStartupFailure:
    def __init__(self) -> None:
        self.logger = Mock()
        self.close = Mock()

    def __iter__(
        self,
    ) -> Iterator[tuple[str, _FailingDeployment | _PendingDeployment]]:
        return iter(
            [
                ("mock_executor_h", _PendingDeployment()),
                ("mock_executor_a", _FailingDeployment()),
            ]
        )


def test_flow_logs_readiness_failure_details(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_WORKFLOW", "1")
    flow = _FlowWithStartupFailure()

    with pytest.raises(RuntimeFailToStart):
        Flow._wait_until_all_ready(flow)

    failure = "OSError(98, 'Address already in use')"
    flow.logger.error.assert_any_call(
        "Deployment %r failed readiness: %s",
        "mock_executor_a",
        failure,
        exc_info=True,
    )
    flow.logger.error.assert_any_call(
        "Flow startup failed. Failed deployments: %s; "
        "pending or cancelled deployments: %s",
        {"mock_executor_a": failure},
        ["mock_executor_h"],
    )
    flow.close.assert_called_once_with()
