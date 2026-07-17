from dataclasses import dataclass

from openinference.semconv.trace import SpanAttributes

from marie.instrumentation import MarieSpanAttributes, set_llm_io


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


@dataclass
class MediaContext:
    ref_id: str
    ref_type: str
    page_number: int


def test_inline_media_is_redacted_without_a_host_resolver() -> None:
    span = RecordingSpan()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,secret"},
                }
            ],
        }
    ]

    set_llm_io(span, input_messages=messages)

    input_value = str(span.attributes[SpanAttributes.INPUT_VALUE])
    assert "data:image/png" not in input_value
    assert "marie://unresolved-media" in input_value
    assert (
        span.attributes[MarieSpanAttributes.media_reference_mode("input")]
        == "unresolved"
    )


def test_host_resolver_replaces_inline_media_reference() -> None:
    span = RecordingSpan()
    context = MediaContext(ref_id="document.tif", ref_type="claim", page_number=2)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,secret"},
                }
            ],
        }
    ]

    set_llm_io(
        span,
        input_messages=messages,
        context=context,
        media_reference_resolver=lambda value: (
            f"s3://documents/{value.ref_id}",
            "s3_asset_path",
        ),
    )

    input_value = str(span.attributes[SpanAttributes.INPUT_VALUE])
    assert "data:image/png" not in input_value
    assert "s3://documents/document.tif" in input_value
    assert (
        span.attributes[MarieSpanAttributes.media_reference_mode("input")]
        == "s3_asset_path"
    )
