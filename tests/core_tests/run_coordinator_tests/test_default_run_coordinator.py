from typing import Iterator

import pytest
from marie._core.host_representation.external import ExternalJob
from marie._core.instance import DagsterInstance
from marie._core.run_coordinator import SubmitRunContext
from marie._core.run_coordinator.base import RunCoordinator
from marie._core.run_coordinator.default_run_coordinator import DefaultRunCoordinator
from marie._core.storage.marie_run import DagsterRun, DagsterRunStatus
from marie._core.test_utils import create_run_for_test, instance_for_test
from marie.utils.merger import merge_dicts


@pytest.fixture()
def instance() -> Iterator[DagsterInstance]:
    overrides = {
        "run_launcher": {"module": "dagster._core.test_utils", "class": "MockedRunLauncher"}
    }
    with instance_for_test(overrides=overrides) as inst:
        yield inst


@pytest.fixture()
def coodinator(instance: DagsterInstance) -> Iterator[RunCoordinator]:
    run_coordinator = DefaultRunCoordinator()
    run_coordinator.register_instance(instance)
    yield run_coordinator


def test_submit_run(instance: DagsterInstance, coodinator: DefaultRunCoordinator):
    with get_bar_workspace(instance) as workspace:
        external_job = (
            workspace.get_code_location("bar_code_location")
            .get_repository("bar_repo")
            .get_full_external_job("foo")
        )

        run = _create_run(instance, external_job, run_id="foo-1")
        returned_run = coodinator.submit_run(SubmitRunContext(run, workspace))
        assert returned_run.run_id == "foo-1"
        assert returned_run.status == DagsterRunStatus.STARTING

        assert len(instance.run_launcher.queue()) == 1  # type: ignore
        assert instance.run_launcher.queue()[0].run_id == "foo-1"  # type: ignore
        assert instance.run_launcher.queue()[0].status == DagsterRunStatus.STARTING  # type: ignore
        assert instance.get_run_by_id("foo-1")
