# marie-instrumentation

`marie-instrumentation` is Marie's reusable observability abstraction over
OpenTelemetry and OpenInference. It provides span helpers and preserves the
tracker API used by existing Marie runtimes without depending on the `marie-ai`
server package.

The distribution name is `marie-instrumentation`; its PEP 420 import surface
is `marie.instrumentation`.

```python
from marie.instrumentation import configure, get_tracker, register

register(project_name="document-processing", service_name="extract-executor")
configure({"enabled": True, "exporter": "otel"})

tracker = get_tracker()
with tracker.trace("extract") as trace:
    generation_id = tracker.generation(
        trace_id=trace.id,
        name="completion",
        model="gpt-4o",
        input=[{"role": "user", "content": "Extract the document"}],
    )
    tracker.end(generation_id, output="done")
```

Host applications can pass a media reference resolver to `set_llm_io()` when
inline image data should be represented by a durable source URL in telemetry.
The package never imports host storage or request-context implementations.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```
