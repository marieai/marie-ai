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
