import pytest

from marie.utils.error import serialize_error


@pytest.mark.parametrize(
    ("return_data", "expected_type", "expected_message"),
    [
        (None, "RuntimeError", "request failed"),
        (
            {
                "error": ("legacy error",),
                "error_details": {
                    "type": "ContextWindowExceededError",
                    "message": "maximum context length exceeded",
                },
            },
            "ContextWindowExceededError",
            "maximum context length exceeded",
        ),
        (
            {"error": ["first error", "second error"]},
            "RuntimeError",
            "first error; second error",
        ),
        (
            {"error": "legacy error", "error_details": {"message": 42}},
            "RuntimeError",
            "legacy error",
        ),
        (
            {"error": "legacy error", "error_details": "invalid"},
            "RuntimeError",
            "legacy error",
        ),
    ],
)
def test_serialize_error_uses_returned_error_details(
    return_data, expected_type, expected_message
):
    details = serialize_error(
        None,
        return_data,
        default_message="request failed",
    )

    assert details == {
        "type": expected_type,
        "message": expected_message,
        "filename": "unknown",
        "name": "unknown",
        "line_no": 0,
    }


def test_serialize_error_prefers_exception_and_captures_deepest_frame():
    def raise_error():
        raise ValueError("invalid request")

    try:
        raise_error()
    except ValueError as error:
        details = serialize_error(
            error,
            {
                "error_details": {
                    "type": "ReturnedError",
                    "message": "returned message",
                }
            },
            default_message="request failed",
        )

    assert details["type"] == "ValueError"
    assert details["message"] == "invalid request"
    assert details["filename"] == "test_error.py"
    assert details["name"] == "raise_error"
    assert details["line_no"] > 0


def test_serialize_error_omits_nested_diagnostics_by_default():
    try:
        try:
            raise ValueError("internal downstream details")
        except ValueError as cause:
            raise RuntimeError("request failed") from cause
    except RuntimeError as error:
        details = serialize_error(error, default_message="request failed")

    assert set(details) == {"type", "message", "filename", "name", "line_no"}
    assert details["filename"] == "test_error.py"
    assert "internal downstream details" not in str(details)


def test_serialize_error_omits_returned_diagnostics_by_default():
    details = serialize_error(
        None,
        {
            "error_details": {
                "type": "RuntimeError",
                "message": "request failed",
                "filename": "/internal/worker/task.py",
                "traceback": "internal traceback",
                "cause": {"message": "internal downstream details"},
                "failed_tasks": [{"error": {"message": "internal task details"}}],
            }
        },
        default_message="request failed",
    )

    assert details == {
        "type": "RuntimeError",
        "message": "request failed",
        "filename": "unknown",
        "name": "unknown",
        "line_no": 0,
    }


def test_serialize_error_can_silence_returned_message():
    details = serialize_error(
        None,
        {
            "error_details": {
                "type": "ContextWindowExceededError",
                "message": "maximum context length exceeded",
            }
        },
        default_message="request failed",
        silence_exceptions=True,
    )

    assert details["type"] == "ContextWindowExceededError"
    assert details["message"] == "request failed"


def test_serialize_error_keeps_batch_identity_and_original_exception_traceback():
    from marie.engine.batch_processor import BatchResult
    from marie.engine.exceptions import BatchExecutionError
    from marie.engine.llm_queue.producer import QueueTaskError

    try:
        raise BatchExecutionError(
            request_id="request-a",
            failed_results=[
                BatchResult(
                    "task-a",
                    None,
                    QueueTaskError(
                        "call_timeout", state="outcome_unknown", confirmed=True
                    ),
                )
            ],
            total=1,
        )
    except BatchExecutionError as error:
        details = serialize_error(
            error, default_message="request failed", include_diagnostics=True
        )
        silenced_exception = serialize_error(
            error,
            default_message="request failed",
            silence_exceptions=True,
            include_diagnostics=True,
        )
    assert set(silenced_exception) == {"type", "message", "filename", "name", "line_no"}
    assert silenced_exception["message"] == "request failed"
    assert details["request_id"] == "request-a"
    assert details["primary_task_id"] == "task-a"
    assert details["failed_count"] == 1
    assert details["failed_tasks"][0]["error"]["category"] == "call_timeout"
    assert details["failed_tasks"][0]["error"]["state"] == "outcome_unknown"
    assert "BatchExecutionError" in details["traceback"]
    persisted = serialize_error(
        None,
        {"error_details": details},
        default_message="request failed",
        include_diagnostics=True,
    )
    assert persisted["traceback"] == details["traceback"]
    assert persisted["failed_tasks"] == details["failed_tasks"]
    silenced = serialize_error(
        None,
        {"error_details": details},
        default_message="request failed",
        silence_exceptions=True,
        include_diagnostics=True,
    )
    assert set(silenced) == {"type", "message", "filename", "name", "line_no"}
    assert silenced["message"] == "request failed"


def test_serialize_error_handles_batch_records_without_an_exception():
    from marie.engine.exceptions import BatchExecutionError

    error = BatchExecutionError("request-a", [object()], total=1)
    details = serialize_error(
        error, default_message="request failed", include_diagnostics=True
    )
    assert details["failed_count"] == 1
    assert details["failed_tasks"] == []


def test_serialize_error_marks_truncated_tracebacks():
    try:
        raise ValueError("x" * 80_000)
    except ValueError as error:
        details = serialize_error(
            error, default_message="request failed", include_diagnostics=True
        )
    assert details["traceback_truncated"] is True
    assert len(details["traceback"].encode("utf-8")) <= 65_536
