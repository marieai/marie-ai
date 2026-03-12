import pytest

from marie.logging_core.logger import MarieLogger as JinaLogger


@pytest.fixture()
def logger():
    return JinaLogger("test networking")
