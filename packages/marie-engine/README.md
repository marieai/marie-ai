# Marie Engine

`marie-engine` is the reusable model-execution layer for the Marie ecosystem.
It contributes `marie.engine` to the implicit PEP 420 `marie` namespace and can
be installed without the `marie-ai` server distribution.

```bash
uv add 'marie-engine[openai,agent]'
```

```python
from marie.engine import EngineLM, get_engine
from marie.engine.agent_wrapper import MarieEngineLLMWrapper
```

The base package owns engine contracts, provider selection, completion
contracts, OpenAI-compatible execution, queue primitives, instrumentation, and
the optional bridge to `marie-agent`. Heavy provider dependencies are extras.

Marie AI retains server concerns such as PostgreSQL scheduler configuration,
runtime startup, credentials, storage-backed media resolution, Gateway
lifecycle, and deployment routing. Hosts supply those concerns through the
portable contracts rather than the engine importing the server.

`marie-engine` may depend on `marie-agent` through its `agent` extra. The
provider-independent `marie-agent` package does not depend on `marie-engine`,
which keeps the dependency graph acyclic.

## Namespace contract

The wheel contains `marie/engine` and deliberately does not contain
`marie/__init__.py`. Do not add a namespace-root initializer.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv build
```
