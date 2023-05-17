import sys
from contextlib import ExitStack, contextmanager
from typing import Iterator, Optional

# from dagster import file_relative_path
# from marie._core.host_representation import (
#     JobHandle,
#     ManagedGrpcPythonEnvCodeLocationOrigin,
# )
# from marie._core.host_representation.handle import RepositoryHandle
from marie._core.instance import DagsterInstance
from marie._core.test_utils import instance_for_test
from marie._core.workspace.context import WorkspaceProcessContext, WorkspaceRequestContext
# from marie._core.workspace.load_target import PythonFileTarget


def file_relative_path(__file__, param):
    pass


@contextmanager
def get_bar_workspace(instance: DagsterInstance) -> Iterator[WorkspaceRequestContext]:
    with WorkspaceProcessContext(
        instance,
        PythonFileTarget(
            python_file=file_relative_path(__file__, "api_tests_repo.py"),
            attribute="bar_repo",
            working_directory=None,
            location_name="bar_code_location",
        ),
    ) as workspace_process_context:
        yield workspace_process_context.create_request_context()
