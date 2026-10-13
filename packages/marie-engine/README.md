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

## Timeouts

OpenAI-compatible requests wait up to 600 seconds for a response, and a batch
has a 900-second deadline. Override these defaults when needed:

```bash
LLM_HTTP_READ_TIMEOUT_S=600
LLM_BATCH_TIMEOUT_S=900
```

Keep the batch timeout greater than the HTTP read timeout so queued work has
time to acquire a request slot and complete.

## External model configuration

`marie.registry` owns model identity, URI resolution, metadata, capability
inference, and complete deployment profiles. The engine contains no model
catalog, model-specific configuration functions, or dispatch table.

Marie-AI reads `models.yaml` from its configured directory, such as
`/mnt/data/marie-ai/config/models.yaml`. Standalone hosts can set
`MARIE_DEFAULT_MOUNT` or `MARIE_MODEL_CATALOG` before model use. Generic imports
work without a catalog; model resolution requires the configured file.

Profiles supply loading and quantization policy, runtime limits and GPU budget,
processor and image settings, prompt templates, sampling defaults, cache policy,
and provider availability. Hugging Face or local metadata supplies capability
and available generation defaults. Explicit deployment values take precedence.

```python
engine = get_engine(
    'document_vision',
    provider='vllm',
    runtime_kwargs={'max_model_len': 16384},
    processor_kwargs={'max_pixels': 1003520},
    generation_kwargs={'max_tokens': 512},
)
```

Constructor overrides win over model/provider profiles. Requests can override
sampling and supported image preparation settings. Adding a supported model or
changing tuning values requires external configuration. Existing MODEL_NAME_MAP,
REVERSE_MODEL_NAME_MAP, and __MULTIMODAL_MODELS__ dict/list APIs remain available
through lazy registry-backed lookup.

See [Marie Registry](../marie-registry/README.md) for the versioned schema,
precedence, metadata resolution, quantizer fallback, and standalone setup.
Declaring a model does not install it or add support to the selected provider.

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
