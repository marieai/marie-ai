from __future__ import annotations


def test_create_otlp_span_exporter_uses_http_protocol(monkeypatch):
    from marie import instrumentation

    created: dict[str, object] = {}

    class FakeHttpExporter:
        def __init__(self, *, endpoint: str):
            created["protocol"] = "http/protobuf"
            created["endpoint"] = endpoint

    class FakeGrpcExporter:
        def __init__(self, *, endpoint: str, insecure: bool):
            created["protocol"] = "grpc"
            created["endpoint"] = endpoint
            created["insecure"] = insecure

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        FakeHttpExporter,
    )
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter",
        FakeGrpcExporter,
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    exporter = instrumentation._create_otlp_span_exporter("http://localhost:4318")

    assert isinstance(exporter, FakeHttpExporter)
    assert created == {
        "protocol": "http/protobuf",
        "endpoint": "http://localhost:4318/v1/traces",
    }


def test_create_otlp_span_exporter_defaults_to_grpc(monkeypatch):
    from marie import instrumentation

    created: dict[str, object] = {}

    class FakeGrpcExporter:
        def __init__(self, *, endpoint: str, insecure: bool):
            created["endpoint"] = endpoint
            created["insecure"] = insecure

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter",
        FakeGrpcExporter,
    )
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)

    exporter = instrumentation._create_otlp_span_exporter("http://localhost:4317")

    assert isinstance(exporter, FakeGrpcExporter)
    assert created == {
        "endpoint": "http://localhost:4317",
        "insecure": True,
    }


def test_grpc_trace_exporter_uses_large_message_channel_options(monkeypatch):
    from marie import instrumentation

    created: dict[str, object] = {}

    class FakeGrpcExporter:
        def __init__(
            self,
            *,
            endpoint: str,
            insecure: bool,
            channel_options: tuple[tuple[str, int], ...],
        ):
            created["endpoint"] = endpoint
            created["insecure"] = insecure
            created["channel_options"] = channel_options

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter",
        FakeGrpcExporter,
    )
    monkeypatch.setenv("MARIE_OTEL_GRPC_MAX_MESSAGE_BYTES", str(64 * 1024 * 1024))

    exporter = instrumentation._create_otlp_span_exporter("http://localhost:4317")

    assert isinstance(exporter, FakeGrpcExporter)
    assert created == {
        "endpoint": "http://localhost:4317",
        "insecure": True,
        "channel_options": (
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ),
    }


def test_trace_batch_processor_defaults_to_stress_safe_export_batch(monkeypatch):
    from marie import instrumentation

    monkeypatch.delenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", raising=False)

    assert instrumentation._batch_span_processor_kwargs() == {
        "max_export_batch_size": 32
    }


def test_trace_batch_processor_respects_export_batch_env(monkeypatch):
    from marie import instrumentation

    monkeypatch.setenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "8")

    assert instrumentation._batch_span_processor_kwargs() == {
        "max_export_batch_size": 8
    }


def test_trace_specific_otlp_protocol_overrides_general_protocol(monkeypatch):
    from marie import instrumentation

    created: dict[str, object] = {}

    class FakeHttpExporter:
        def __init__(self, *, endpoint: str):
            created["protocol"] = "http/protobuf"
            created["endpoint"] = endpoint

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        FakeHttpExporter,
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")

    exporter = instrumentation._create_otlp_span_exporter("http://localhost:4318")

    assert isinstance(exporter, FakeHttpExporter)
    assert created == {
        "protocol": "http/protobuf",
        "endpoint": "http://localhost:4318/v1/traces",
    }


def test_http_trace_endpoint_path_is_not_appended_twice(monkeypatch):
    from marie import instrumentation

    created: dict[str, object] = {}

    class FakeHttpExporter:
        def __init__(self, *, endpoint: str):
            created["endpoint"] = endpoint

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        FakeHttpExporter,
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    exporter = instrumentation._create_otlp_span_exporter(
        "http://localhost:4318/v1/traces"
    )

    assert isinstance(exporter, FakeHttpExporter)
    assert created == {"endpoint": "http://localhost:4318/v1/traces"}
