"""Timeout utilities for ETCD operations."""

import threading
from typing import Callable, TypeVar

T = TypeVar('T')


class OperationTimeoutError(Exception):
    """Raised when an operation exceeds its timeout."""

    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Operation '{operation}' timed out after {timeout}s")


def run_with_timeout(
    func: Callable[..., T],
    timeout: float,
    operation_name: str = "operation",
) -> T:
    """
    Execute a function with a hard wall-clock timeout.

    Runs func on a daemon thread and joins with the timeout. On timeout the
    caller is released immediately and the worker thread is abandoned (it
    parks on the underlying call; being a daemon it never blocks process
    exit). Exceptions from func are re-raised in the caller.
    """
    result: list = []
    error: list = []

    def _target():
        try:
            result.append(func())
        except BaseException as e:  # noqa: BLE001 - must ferry everything back
            error.append(e)

    t = threading.Thread(target=_target, daemon=True, name=f"timeout-{operation_name}")
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise OperationTimeoutError(operation_name, timeout)
    if error:
        raise error[0]
    return result[0]
