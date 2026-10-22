from __future__ import annotations


def _service_name(**kwargs) -> str:
    from marie.instrumentation import register

    provider = register(set_global_tracer_provider=False, **kwargs)
    return provider.resource.attributes["service.name"]


def test_register_defaults_service_name_to_marie_ai(monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert _service_name() == "marie-ai"


def test_register_uses_otel_service_name_env(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "marie-ai-vagrant")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert _service_name() == "marie-ai-vagrant"
