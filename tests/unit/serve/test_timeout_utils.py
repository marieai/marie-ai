import time

import pytest

from marie.serve.discovery.timeout_utils import (
    OperationTimeoutError,
    run_with_timeout,
)


def test_returns_result():
    assert run_with_timeout(lambda: 42, timeout=1.0) == 42


def test_reraises_original_exception():
    with pytest.raises(ValueError, match="boom"):
        run_with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), timeout=1.0)


def test_timeout_bounds_wall_clock():
    start = time.monotonic()
    with pytest.raises(OperationTimeoutError):
        run_with_timeout(lambda: time.sleep(5), timeout=0.3, operation_name="hang")
    elapsed = time.monotonic() - start
    # the whole point: the caller is released at ~timeout, not after func returns
    assert elapsed < 1.5
