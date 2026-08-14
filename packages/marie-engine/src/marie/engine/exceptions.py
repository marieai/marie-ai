"""Exceptions raised by reusable Marie engine operations."""

from typing import Any


class MaxTokensExceededError(Exception):
    """Raised when an LLM stops because it reached the token limit."""

    def __init__(self, message: str = "LLM hit max_tokens") -> None:
        super().__init__(message)


class RepetitionError(Exception):
    """Raised when an LLM response becomes repetitious."""

    def __init__(self, message: str = "LLM output is repetitive") -> None:
        super().__init__(message)


class CircuitOpenError(Exception):
    """Raised when calls to a backend are being shed by its circuit breaker."""

    def __init__(self, address: str, message: str = "") -> None:
        self.address = address
        super().__init__(message or f"Circuit breaker open for {address}")


class BatchExecutionError(Exception):
    """Raised when one or more tasks in an engine batch fail."""

    def __init__(
        self,
        request_id: str,
        failed_results: list[Any],
        total: int,
        message: str = "",
    ) -> None:
        self.request_id = request_id
        self.failed_results = failed_results
        self.total = total
        primary_result = next(
            (
                result
                for result in failed_results
                if getattr(result, "error", None) is not None
            ),
            None,
        )
        self.primary_error: Exception | None = (
            primary_result.error if primary_result is not None else None
        )
        self.primary_task_id: str | None = (
            primary_result.task_id if primary_result is not None else None
        )
        failed_count = len(failed_results)
        detail = (
            f"Batch inference failed: {failed_count}/{total} tasks failed "
            f"(request_id={request_id})"
        )
        if self.primary_error is not None:
            detail += (
                f"; first failure task_id={self.primary_task_id}: "
                f"{type(self.primary_error).__name__}: {self.primary_error}"
            )
        super().__init__(message or detail)
