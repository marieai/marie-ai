# marie-wasm

Wasmtime-based runtime for executing Marie workflow nodes in WebAssembly.

## Overview

`marie-wasm` provides a secure, sandboxed execution environment for Marie workflow nodes using the WebAssembly Component Model. It supports compiling user code from multiple languages (Rust, Python, JavaScript) into Wasm components that can be executed with fine-grained permission controls.

## Features

- **Multi-language support**: Compile Rust, Python, and JavaScript to Wasm components
- **Docker-based compilation**: Secure, isolated compilation in containers
- **Component Model**: Uses WASI Preview 2 and WIT interfaces
- **Permission system**: Fine-grained capability-based security
- **Host functions**: HTTP client, secrets, key-value store, logging, events

## Installation

```bash
pip install marie-wasm

# With S3 storage support
pip install marie-wasm[s3]

# For development
pip install marie-wasm[dev]
```

## Quick Start

### Compiling User Code

```python
from marie_wasm import WasmCompilerService, Language

compiler = WasmCompilerService(storage_client=my_storage)

# Compile Python code to Wasm
wasm_path = await compiler.compile(
    code='''
def execute(input_items, config, context):
    results = []
    for item in input_items:
        data = json.loads(item["json"])
        results.append({"json": json.dumps({"processed": data})})
    return {"success": results}
''',
    language=Language.PYTHON,
    node_id="my-node-123",
)
```

### Executing Wasm Modules

The `WasmNodeExecutor` in marie-ai proper handles execution:

```python
from marie.executor.wasm import WasmNodeExecutor

executor = WasmNodeExecutor()
result = await executor.execute(
    docs=[WasmInputDoc(json='{"input": "data"}')],
    parameters={"wasm_path": wasm_path},
)
```

## Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  marie-studio   │      │     Gateway     │      │ WasmNodeExecutor│
│  (UI Editor)    │─────▶│  (Compilation)  │─────▶│  (Execution)    │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        │
        │ User writes code       │ Compiles to .wasm      │ Loads & runs
        │ in Code node           │ Stores in S3/storage   │ pre-compiled
```

## WIT Interface

All nodes implement the `marie:node@1.0.0` world:

```wit
world node {
    import http-client;
    import secrets;
    import key-value;
    import logging;
    import events;

    export execute: func(
        input: list<data-item>,
        config: config,
        ctx: context
    ) -> execution-result;
}
```

## Supported Languages

| Language | Compiler Tool | Output Size |
|----------|--------------|-------------|
| Rust | cargo-component | ~100KB |
| Python | componentize-py | ~15MB |
| JavaScript | jco + componentize-js | ~3MB |

## Security

### Compiler Containers
- Non-root user execution
- Read-only root filesystem
- Network disabled during compilation
- Memory and CPU limits
- No capabilities granted

### Runtime Sandbox
- Fuel metering for CPU limits
- Epoch-based timeouts
- Memory limits via Wasmtime
- Capability-based permissions
- Host validation for HTTP requests

## Configuration

### Permissions

```python
from marie_wasm import Permissions

perms = Permissions(
    allow_http=True,
    http_allowed_hosts=["api.example.com"],
    allow_secrets=True,
    secret_allowed_names=["API_KEY"],
    allow_kv=True,
    kv_prefix="user-123:",
    max_memory_mb=64,
    max_fuel=1_000_000_000,
    timeout_ms=30_000,
)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/marie_wasm

# Formatting
black src tests
isort src tests
```

## License

Apache 2.0
