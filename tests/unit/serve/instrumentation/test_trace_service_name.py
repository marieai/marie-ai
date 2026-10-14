from unittest.mock import patch

from opentelemetry.trace import ProxyTracerProvider

from marie.serve.instrumentation import (
    EXECUTOR_SERVICE_NAME,
    GATEWAY_SERVICE_NAME,
    InstrumentationMixin,
)
from marie.serve.runtimes.gateway.request_handling import GatewayRequestHandler
from marie.serve.runtimes.servers import _trace_service_name
from marie.serve.runtimes.worker.request_handling import WorkerRequestHandler


def test_gateway_and_executor_report_distinct_service_names():
    assert _trace_service_name(GatewayRequestHandler) == GATEWAY_SERVICE_NAME
    assert _trace_service_name(WorkerRequestHandler) == EXECUTOR_SERVICE_NAME
    assert GATEWAY_SERVICE_NAME != EXECUTOR_SERVICE_NAME


def test_setup_instrumentation_passes_service_name_to_register():
    with patch("marie.instrumentation.register") as register:
        InstrumentationMixin()._setup_instrumentation(
            name="gateway/rep-0", tracing=True, service_name=GATEWAY_SERVICE_NAME
        )

    assert register.call_args.kwargs["service_name"] == GATEWAY_SERVICE_NAME


def test_ensure_tracer_provider_passes_service_name_to_register():
    from marie.utils.server_runtime import _ensure_tracer_provider

    with (
        patch(
            "opentelemetry.trace.get_tracer_provider",
            return_value=ProxyTracerProvider(),
        ),
        patch("marie.instrumentation.register") as register,
    ):
        _ensure_tracer_provider(service_name=EXECUTOR_SERVICE_NAME)

    assert register.call_args.kwargs["service_name"] == EXECUTOR_SERVICE_NAME
