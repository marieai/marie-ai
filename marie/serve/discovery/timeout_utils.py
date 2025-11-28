"""Timeout utilities for ETCD operations."""

import concurrent.futures
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
    Execute a function with a timeout.

    Args:
        func: Zero-argument callable to execute
        timeout: Maximum time in seconds
        operation_name: Name for error messages

    Returns:
        Result of func()

    Raises:
        OperationTimeoutError: If timeout exceeded
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise OperationTimeoutError(operation_name, timeout)
