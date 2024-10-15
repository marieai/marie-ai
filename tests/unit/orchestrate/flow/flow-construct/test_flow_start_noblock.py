import time

import pytest

from marie import Flow
from marie.excepts import RuntimeFailToStart
from marie.logging_core.profile import TimeContext
from marie.serve.executors import BaseExecutor


class SlowExecutor(BaseExecutor):
    def post_init(self):
        time.sleep(4)


@pytest.mark.slow
def test_flow_slow_executor_intra():
    f = Flow().add(uses="SlowExecutor", shards=2)

    with f, TimeContext("start flow") as tc:
        assert tc.now() < 8


@pytest.mark.slow
def test_flow_slow_executor_inter():
    f = Flow().add(uses="SlowExecutor", shards=3).add(uses="SlowExecutor", shards=3)

    with f, TimeContext("start flow") as tc:
        assert tc.now() < 8


@pytest.mark.slow
def test_flow_slow_executor_bad_fail_early():
    f = Flow().add(uses="SlowExecutor", shards=3).add(uses="BADNAME_EXECUTOR", shards=3)

    with pytest.raises(RuntimeFailToStart):
        with f, TimeContext("start flow") as tc:
            assert tc.now() < 8
