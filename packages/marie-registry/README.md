# Marie Registry

`marie-registry` owns the complete `marie.registry` package: the existing model
registry and URI handlers, model metadata, external profiles, and configuration
resolution. It has no dependency on the engine or server distribution.

## Configure deployment paths

Marie-AI initializes the registry after loading runtime configuration:

```python
from marie.registry import ModelRegistry

ModelRegistry.configure(
    config_dir='/mnt/data/marie-ai/config',
    model_root='/mnt/data/marie-ai/model_zoo',
)
config = ModelRegistry.resolve_config('document_vision', 'vllm')
```

Standalone callers can also use `MARIE_DEFAULT_MOUNT` or `MARIE_MODEL_CATALOG`.
Catalog precedence is `MARIE_MODEL_CATALOG`, explicit `catalog_path`, configured
`config_dir/models.yaml`, then `MARIE_DEFAULT_MOUNT/config/models.yaml`.
The existing per-call `__model_path__` overrides the configured native model root.

Generic imports do not read configuration or contact Hugging Face. First model
use loads the selected file. Missing files report their path. Restart after
changing the file or model metadata; changing configured paths invalidates caches.

## Define complete external profiles

The runtime catalog belongs to deployment configuration, such as
`/mnt/data/marie-ai/config/models.yaml`. No models or tuning defaults are bundled
in the package.

```yaml
schema_version: 1
defaults:
  vllm:
    loading:
      dtype: bfloat16
      quantization: null
      load_format: auto
    runtime:
      max_model_len: 8192
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.85
    image:
      min_pixels: 784
      max_pixels: 1003520
    generation:
      max_tokens: 4096
      temperature: 0
      top_p: 1.0
      top_k: -1
      n: 1
      stop_token_ids: null
      structured_outputs:
        disable_any_whitespace: true
    prompt:
      system: You are a helpful assistant for processing documents.
      template: null
    cache: false
profiles:
  vision:
    vllm:
      runtime:
        max_model_len: 16384
      processor:
        fps: 1
models:
  document_vision:
    name: Qwen/Qwen3-VL-4B-Instruct
    profile: vision
  private_vision:
    name: zoo://private-vision
    multimodal: true
    profile: vision
  disabled_model:
    name: your-org/disabled-model
    providers:
      vllm:
        enabled: false
```

Provider sections include loading, runtime, processor, image preparation,
generation, prompt, cache, and enabled status. A model selects one profile and
can override its settings under `providers`. Dictionary values merge by key;
scalars and lists replace inherited values, including false, null, and empty lists.
Processor overrides also reach image preparation; `image` permits separate image
preparation controls while preserving provider constructor arguments.

Resolution applies metadata generation defaults, deployment provider defaults,
the selected profile, model overrides, and explicit constructor overrides in that
order. Requests override sampling and image preparation and pass processor arguments
on vLLM prompt dictionaries without rebuilding the model. Image pixel bounds
are applied to prepared PIL images before provider processing. Resolved results are independent copies.

Loading values are provider constructor arguments. An optional `requires` module
and `fallback` mapping define dependency-based loading behavior in data:

```yaml
loading:
  quantization: bitsandbytes
  load_format: bitsandbytes
  dtype: bfloat16
  requires: bitsandbytes
  fallback:
    quantization: null
    load_format: auto
    dtype: bfloat16
```

Use `requires: null` and `fallback: null` when overriding this policy with a
loading mode that does not depend on that module. Provider adapters validate
against the installed runtime; quantization and model dtype remain distinct.

## Resolve metadata and artifacts

`ModelRegistry.get`, `config`, `checkpoint`, `discover`, and registered URI handlers
retain their APIs. Native schemes include file/model/zoo; Hugging Face schemes
include hf/transformer. Direct supported model identifiers work without an alias.

`ModelRegistry.metadata` reads model, processor, tokenizer, and generation JSON
metadata without loading weights or executing remote model code. Hugging Face
reads only those metadata files, using the Hub cache and offline settings.
Missing remote optional metadata files are tolerated. Explicit offline mode
also tolerates uncached optional files; normal transport and access errors remain
errors. The handler's observed commit propagates to artifacts and provider
loading; the engine reuses the provider tokenizer.

Omit `multimodal` to infer image support from metadata and upstream Transformers
mappings. Supply true/false for a custom model or endpoint. Unrecognized metadata
requires an explicit override. Conflicting overrides, revisions, ambiguous
canonical references, unknown profiles, duplicate keys, and invalid settings fail
with configuration context.

Install `marie-registry[huggingface]` for Hub downloads and Transformers task
mappings. Imports and explicit capability checks do not require that extra. Resolving a
complete enabled provider config always reads metadata through its URI handler,
even when capability is explicitly configured. Native custom models can supply
`marie.json` locally.

## Verify

From the Marie-AI root, use the environment and package source roots for the
registry tests. Build the independent wheel from this directory with `uv build`.
The wheel owns `marie/registry` and has no `marie/__init__.py`, server code, engine
code, or bundled deployment catalog.
