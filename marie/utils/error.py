import traceback
from types import TracebackType
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Type, Union

from typing_extensions import TypeAlias

import marie.check as check
from marie.excepts import BaseMarieException, BatchExecutionError


# mypy does not support recursive types, so "cause" has to be typed `Any`
# @whitelist_for_serdes
class SerializableErrorInfo(
    NamedTuple(
        "SerializableErrorInfo",
        [
            ("message", str),
            ("stack", Sequence[str]),
            ("cls_name", Optional[str]),
            ("cause", Any),
            ("context", Any),
        ],
    )
):
    # serdes log
    # * added cause - default to None in constructor to allow loading old entries
    # * added context - default to None for similar reasons
    #
    def __new__(
        cls,
        message: str,
        stack: Sequence[str],
        cls_name: Optional[str],
        cause: Optional["SerializableErrorInfo"] = None,
        context: Optional["SerializableErrorInfo"] = None,
    ):
        return super().__new__(cls, message, stack, cls_name, cause, context)

    def __str__(self) -> str:
        return self.to_string()

    def to_string(self) -> str:
        stack_str = "\nStack Trace:\n" + "".join(self.stack) if self.stack else ""
        cause_str = (
            "\nThe above exception was caused by the following exception:\n"
            + self.cause.to_string()
            if self.cause
            else ""
        )
        context_str = (
            "\nThe above exception occurred during handling of the following exception:\n"
            + self.context.to_string()
            if self.context
            else ""
        )

        return "{err.message}{stack}{cause}{context}".format(
            err=self, stack=stack_str, cause=cause_str, context=context_str
        )

    def to_exception_message_only(self) -> "SerializableErrorInfo":
        """Return a new SerializableErrorInfo with only the message and cause set.

        This is done in cases when the context about the error should not be exposed to the user.
        """
        return SerializableErrorInfo(
            message=self.message, stack=[], cls_name=self.cls_name
        )


def _serializable_error_info_from_tb(
    tb: traceback.TracebackException,
) -> SerializableErrorInfo:
    return SerializableErrorInfo(
        # usually one entry, multiple lines for SyntaxError
        "".join(list(tb.format_exception_only())),
        tb.stack.format(),
        tb.exc_type.__name__ if tb.exc_type is not None else None,
        _serializable_error_info_from_tb(tb.__cause__) if tb.__cause__ else None,
        _serializable_error_info_from_tb(tb.__context__) if tb.__context__ else None,
    )


ExceptionInfo: TypeAlias = Union[
    Tuple[Type[BaseException], BaseException, TracebackType],
    Tuple[None, None, None],
]


def serialize_error(
    exception: Exception | None,
    return_data: Any = None,
    *,
    default_message: str,
    silence_exceptions: bool = False,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Serialize failure details; internal diagnostics require explicit opt-in."""
    filename = "unknown"
    name = "unknown"
    line_no = 0

    if exception is not None and exception.__traceback__ is not None:
        traceback_head = exception.__traceback__
        traceback_tail = traceback_head
        while traceback_tail.tb_next:
            traceback_tail = traceback_tail.tb_next
        filename = traceback_tail.tb_frame.f_code.co_filename
        name = traceback_tail.tb_frame.f_code.co_name
        line_no = traceback_tail.tb_lineno
        traceback.clear_frames(traceback_head)

    returned_type, returned_message = _returned_error(return_data)
    message = default_message
    if not silence_exceptions:
        if exception is not None:
            message = str(exception)
        elif returned_message:
            message = returned_message

    details: dict[str, Any] = {
        "type": (
            type(exception).__name__
            if exception is not None
            else returned_type or "RuntimeError"
        ),
        "message": message,
        "filename": filename.rsplit("/", 1)[-1],
        "name": name,
        "line_no": line_no,
    }
    if include_diagnostics and not silence_exceptions:
        if exception is not None:
            details.update(_exception_diagnostics(exception))
        elif isinstance(return_data, dict) and isinstance(
            return_data.get("error_details"), dict
        ):
            returned = return_data["error_details"]
            for key in (
                "filename",
                "name",
                "line_no",
                "traceback",
                "traceback_truncated",
                "category",
                "state",
                "confirmed",
                "request_id",
                "primary_task_id",
                "total",
                "failed_count",
                "failed_tasks",
                "failed_tasks_truncated",
                "cause",
                "batch_error_type",
            ):
                if key in returned:
                    details[key] = returned[key]
    return details


def _exception_diagnostics(exception: Exception, depth: int = 0) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if exception.__traceback__ is not None:
        encoded = "".join(traceback.format_exception(exception)).encode("utf-8")
        if len(encoded) > 65_536:
            marker = b"\n[truncated]"
            details["traceback"] = encoded[: 65_536 - len(marker)].decode(
                "utf-8", errors="ignore"
            ) + marker.decode("utf-8")
            details["traceback_truncated"] = True
        else:
            details["traceback"] = encoded.decode("utf-8")
    for key in ("category", "state", "confirmed"):
        value = getattr(exception, key, None)
        if isinstance(value, (str, int, bool)):
            details[key] = value
    if depth >= 4:
        return details
    if isinstance(exception, BatchExecutionError):
        details.update(
            batch_error_type=type(exception).__name__,
            request_id=exception.request_id,
            primary_task_id=exception.primary_task_id,
            total=exception.total,
            failed_count=len(exception.failed_results),
            failed_tasks_truncated=len(exception.failed_results) > 25,
            failed_tasks=[
                {
                    "task_id": getattr(result, "task_id", None),
                    "error": _nested_exception(result.error, depth + 1),
                }
                for result in exception.failed_results[:25]
                if isinstance(getattr(result, "error", None), Exception)
            ],
        )
        cause = exception.primary_error
    else:
        cause = exception.__cause__ or exception.__context__
    if isinstance(cause, Exception) and cause is not exception:
        details["cause"] = _nested_exception(cause, depth + 1)
    return details


def _nested_exception(exception: Exception, depth: int) -> dict[str, Any]:
    return {
        "type": type(exception).__name__,
        "message": str(exception),
        **_exception_diagnostics(exception, depth),
    }


def _returned_error(return_data: Any) -> tuple[str | None, str | None]:
    if not isinstance(return_data, dict):
        return None, None

    returned_type = None
    returned_message = None
    error_details = return_data.get("error_details")
    if isinstance(error_details, dict):
        if isinstance(error_details.get("type"), str):
            returned_type = error_details["type"]
        if isinstance(error_details.get("message"), str):
            returned_message = error_details["message"]

    if returned_message is None:
        raw_error = return_data.get("error")
        if isinstance(raw_error, (list, tuple)):
            returned_message = "; ".join(str(item) for item in raw_error)
        elif raw_error is not None:
            returned_message = str(raw_error)

    return returned_type, returned_message


def serializable_error_info_from_exc_info(
    exc_info: ExceptionInfo,
    # Whether to forward serialized errors thrown from subprocesses
    hoist_user_code_error: Optional[bool] = True,
) -> SerializableErrorInfo:
    # `sys.exc_info() return Tuple[None, None, None] when there is no exception being processed. We accept this in
    # the type signature here since this function is meant to directly receive the return value of
    # `sys.exc_info`, but the function should never be called when there is no exception to process.
    exc_type, e, tb = exc_info
    additional_message = "sys.exc_info() called but no exception available to process."
    exc_type = check.not_none(exc_type, additional_message=additional_message)
    e = check.not_none(e, additional_message=additional_message)
    tb = check.not_none(tb, additional_message=additional_message)

    if (
        hoist_user_code_error
        and isinstance(e, BaseMarieException)
        and len(e.user_code_process_error_infos) == 1
    ):
        return e.user_code_process_error_infos[0]
    else:
        tb_exc = traceback.TracebackException(exc_type, e, tb)
        return _serializable_error_info_from_tb(tb_exc)
