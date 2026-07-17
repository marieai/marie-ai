# Marie AI Packages

This directory contains separately installable packages that are part of the Marie AI ecosystem.

## Available Packages

### 📦 marie-agent
**Reusable agent runtime**

Provider-independent agents, messages, tools, skills, guardrails, coordination,
and optional backend integrations. Marie server adapters depend on this package;
the package does not depend on the Marie server.

- **Purpose**: Share the agent runtime across Marie applications
- **Install**: `uv add marie-agent`
- **Import**: `marie.agent`
- **Docs**: [packages/marie-agent/README.md](./marie-agent/README.md)

### 📦 marie-instrumentation
**Shared OpenTelemetry and OpenInference instrumentation**

The reusable observability layer used by Marie runtimes and separately
installable components. It includes the active tracker-to-OpenTelemetry adapter
without depending on the `marie-ai` server package.

- **Purpose**: Shared LLM tracing, OpenInference attributes, and OTel export
- **Install**: `uv add marie-instrumentation`
- **Import**: `marie.instrumentation`
- **Docs**: [packages/marie-instrumentation/README.md](./marie-instrumentation/README.md)

### 📦 marie-engine
**Reusable model execution engine**

Provider selection, completion contracts, OpenAI-compatible execution, LLM
queue primitives, and the optional `marie-agent` bridge. Server persistence and
Gateway lifecycle remain in Marie AI adapters.

- **Purpose**: Share model execution across Marie AI and agent applications
- **Install**: `uv add 'marie-engine[openai,agent]'`
- **Import**: `marie.engine`
- **Docs**: [packages/marie-engine/README.md](./marie-engine/README.md)

### 📦 marie-cli
**Marie AI command-line frontend**

The separately installable CLI contributes `marie.cli` and the `marie`
console script. It depends on `marie-ai`; the server package does not depend on
the CLI at runtime. The server source retains a thin `marie/__main__.py`
launcher so existing file-based run configurations and `python -m marie`
continue to work when `marie-cli` is installed.

- **Purpose**: Keep command parsing and console UX independently versioned
- **Install**: `uv add marie-cli`
- **Import**: `marie.cli`
- **Commands**: `marie gateway --help` or `python -m marie gateway --help`
- **Docs**: [packages/marie-cli/README.md](./marie-cli/README.md)

### 📦 marie-mcp
**Lightweight MCP server for AI assistant integration**

A Model Context Protocol (MCP) server that enables AI assistants like Claude to interact with Marie AI's document intelligence capabilities.

- **Size**: ~5MB (vs 2-5GB for main marie-ai package)
- **Purpose**: Client-side integration for AI assistants
- **Install**: `uv add marie-mcp`
- **Import**: `marie.mcp.server`
- **Docs**: [packages/marie-mcp/README.md](./marie-mcp/README.md)

**Use cases**:
- Claude Desktop integration
- LangChain agents
- OpenAI Agents SDK
- Custom AI assistant integrations

### 📦 marie-kernel
**State management kernel for DAG task execution**

A state passing system that enables tasks within a DAG run to share state via simple key-value operations.

- **Purpose**: Cross-task state management for workflow orchestration
- **Install**: `uv add marie-kernel` or `uv add 'marie-kernel[postgres]'`
- **Import**: `marie.kernel`
- **Docs**: [packages/marie-kernel/README.md](./marie-kernel/README.md)

**Features**:
- Simple `ctx.set()`/`ctx.get()` API
- Multi-tenant isolation
- PostgreSQL backend for production
- In-memory backend for testing
- Retry-safe with try_number scoping

### 📦 marie-extension
**Extension package schema, loader, and validator**

A metadata-first package contract for Marie extensions. It validates `marie-extension.yaml` packages from local directories or standard ZIP archives without executing package code.

- **Purpose**: Extension authoring schema, safe package loading, and validation
- **Install**: `uv add marie-extension`
- **Import**: `marie.extension`
- **Docs**: [packages/marie-extension/README.md](./marie-extension/README.md)

**Features**:
- YAML-first extension package schema
- Standard ZIP discovery by `marie-extension.yaml`
- Safe path validation for package files
- Deny-by-default permission models

### 📦 marie-mem0
**Mem0-backed agent memory integration**

The reusable memory adapter used by Marie agents without requiring the
`marie-ai` server distribution.

- **Purpose**: Persistent conversational and agent memory through Mem0
- **Install**: `uv add marie-mem0`
- **Import**: `marie.mem0`
- **Docs**: [packages/marie-mem0/README.md](./marie-mem0/README.md)

### 📦 marie-wasm
**Wasmtime workflow-node runtime**

A separately installable runtime for executing Marie workflow nodes compiled
to WebAssembly.

- **Purpose**: Isolated WebAssembly node execution and daemon integration
- **Install**: `uv add marie-wasm`
- **Import**: `marie.wasm`
- **Docs**: [packages/marie-wasm/README.md](./marie-wasm/README.md)

### 📦 marie-plugin-daemon
**Extension runtime daemon for local plugin execution**

A Go service that validates, installs, starts, and invokes Marie extension packages. It materializes the shared Python runtime and streams session frames between Marie and locally managed plugin processes.

- **Purpose**: Extension package lifecycle and invocation boundary
- **Build**: `cd packages/marie-plugin-daemon && go test ./... && go build -o dist/marie-plugin-daemon ./cmd/server`
- **Docs**: [packages/marie-plugin-daemon/README.md](./marie-plugin-daemon/README.md)

**Features**:
- Health and version endpoints
- Marie ZIP/directory package validation and installation
- Isolated Python environments managed with uv
- Streaming invocation through typed session frames

### 📦 marie-plugin-document-extraction
**First-party document extraction system plugin**

An independently locked plugin that combines Docling Slim and MarkItDown behind one capability and extraction contract. It writes document bodies to request-scoped artifacts and returns bounded descriptors through the daemon.

- **Purpose**: Capability-aware semantic extraction for supported document formats
- **Environment**: Independent uv project created by `marie-plugin-daemon`
- **Docs**: [packages/marie-plugin-document-extraction/README.md](./marie-plugin-document-extraction/README.md)

**Features**:
- Docling Slim and MarkItDown provider dispatch
- Input-aware format capabilities
- File-backed extraction artifacts
- Provider and end-to-end `EmbeddedPlugins` tests

## Plugin Stack Layer Map

Four similarly named pieces make up the plugin stack. They sit on opposite sides of a hard boundary and never import each other across it:

| Layer | Distribution | Import name | Runs in | Role |
|---|---|---|---|---|
| Manifest contract | `marie-extension` | `marie.extension` | Marie control plane, build/CI tooling | Validates `marie-extension.yaml` packages (schema, ZIP loading, permissions). Metadata only — never executes plugin code |
| Plugin host | `marie-plugin-daemon` | n/a (Go) | Daemon process | Installs, starts, and invokes plugins; creates their uv environments; embeds and injects the Python runtime |
| Plugin-side runtime | `marie-plugin-runtime` (source lives in `marie-plugin-daemon/python_runtime/`) | `marie_plugins.runtime` | Inside each plugin process | Stdio protocol shim: frames, sessions, heartbeat, test client. Stdlib-only. Daemon-provided in production; dev-only wheel for plugin authors |
| Plugins | `marie-plugin-{name}` | `marie_plugins.{name}` | Inside their own plugin process | Actual plugin logic (e.g. `marie_plugins.document_extraction`) |

Import rules that keep the boundary honest:

- Plugin protocol code imports `marie_plugins.runtime`. Application plugins may
  install explicitly declared reusable namespace distributions such as
  `marie-agent`, `marie-engine`, and `marie-instrumentation`, but never import
  `marie.runtime` or modules supplied only by the `marie-ai` server.
- Host code never imports plugin internals; it validates manifests with `marie.extension` and talks to plugins through the daemon.
- `marie_plugins` is a PEP 420 implicit namespace shared by the runtime and every plugin. Never create `marie_plugins/__init__.py` — a regular package at that name shadows the other half of the namespace.

Disambiguation: `marie.extension/runtime.py` (the manifest's runtime *envelope model* — network policy, resource limits) is unrelated to `marie_plugins.runtime` (the in-process protocol library).

## Monorepo Structure

```
marie-ai/
├── marie/                      # marie-ai namespace portion (server-side)
│   ├── Core processing
│   ├── ML models & executors
│   └── Gateway & scheduler
│
└── packages/                   # Additional packages (client-side)
    ├── marie-agent/            # Reusable agent runtime
    │   ├── src/marie/agent/
    │   ├── pyproject.toml
    │   └── README.md
    ├── marie-mcp/              # MCP server for AI assistants
    │   ├── src/marie/mcp/server/
    │   ├── pyproject.toml      # Separate PyPI package
    │   └── README.md
    ├── marie-instrumentation/  # Shared OTel/OpenInference instrumentation
    │   ├── src/marie/instrumentation/
    │   ├── pyproject.toml
    │   └── README.md
    ├── marie-engine/           # Reusable model execution engine
    │   ├── src/marie/engine/
    │   ├── pyproject.toml
    │   └── README.md
    ├── marie-cli/              # Marie AI command-line frontend
    │   ├── src/marie/cli/
    │   ├── pyproject.toml
    │   └── README.md
    ├── marie-kernel/           # State management kernel
    │   ├── src/marie/kernel/
    │   ├── pyproject.toml      # Separate PyPI package
    │   └── README.md
    ├── marie-extension/        # Extension package schema and validator
    │   ├── src/marie/extension/
    │   ├── pyproject.toml      # Separate PyPI package
    │   └── README.md
    ├── marie-mem0/             # Mem0 agent-memory integration
    │   ├── src/marie/mem0/
    │   ├── pyproject.toml
    │   └── README.md
    ├── marie-wasm/             # Wasmtime workflow-node runtime
    │   ├── src/marie/wasm/
    │   ├── pyproject.toml
    │   └── README.md
    ├── marie-plugin-daemon/    # Go plugin lifecycle and runtime daemon
    │   ├── cmd/server/
    │   ├── internal/
    │   ├── python_runtime/     # marie_plugins.runtime source (published as marie-plugin-runtime);
    │   │                       # nested here because the daemon go:embeds it at build time
    │   └── go.mod
    └── marie-plugin-document-extraction/ # First-party system plugin
        ├── marie_plugins/document_extraction/
        ├── tests/
        ├── pyproject.toml
        └── uv.lock
```

## Development

### Installing from Source

```bash
# Install main Marie AI package
uv sync

# Install MCP package
cd packages/marie-mcp
uv sync

# Install State Kernel package
cd packages/marie-kernel
uv sync --extra dev

# Install and test instrumentation package
cd packages/marie-instrumentation
uv sync --extra dev
uv run pytest

# Install and test agent package
cd packages/marie-agent
uv sync --extra dev
uv run pytest

# Install and test engine package
cd packages/marie-engine
uv sync --extra dev
uv run pytest

# Install and test CLI package
cd packages/marie-cli
uv sync --extra dev
uv run pytest

# Install Extension package
cd packages/marie-extension
uv sync --extra dev

# Install and test the Document Extraction plugin
cd packages/marie-plugin-document-extraction
uv sync --locked
uv run --locked pytest tests/provider_cases.py tests/packaged_protocol.py -q
```

### Publishing

Each package is published independently to PyPI:

```bash
# Publish main package
uv build
uv publish

# Publish MCP package
cd packages/marie-mcp
uv build
uv publish

# Publish State Kernel package
cd packages/marie-kernel
uv build
uv publish

# Publish Engine package
cd packages/marie-engine
uv build
uv publish

# Publish CLI package
cd packages/marie-cli
uv build
uv publish

# Publish Extension package
cd packages/marie-extension
uv build
uv publish

# Build Plugin Daemon package
cd packages/marie-plugin-daemon
go test ./...
go build -o dist/marie-plugin-daemon ./cmd/server

# Build Document Extraction plugin archive
cd packages/marie-plugin-document-extraction
./scripts/package.sh /path/to/output
```

### Shared Tooling

All packages share:
- Code formatting (black, isort)
- Type checking (mypy)
- Testing (pytest)
- CI/CD pipelines

Configuration in root:
- `.github/workflows/` - CI/CD for all packages
- `pyproject.toml` - Root tooling config
- `.pre-commit-config.yaml` - Shared hooks

## Adding New Packages

To add a new package to the monorepo:

1. Create directory: `packages/your-package/`
2. Add `pyproject.toml` with package metadata
3. Create `src/marie/your_package/` without `src/marie/__init__.py`
4. Add README.md with documentation
5. Update this README.md
6. Add CI workflow in `.github/workflows/`

## Package Guidelines

Each package should:
- ✅ Be independently installable
- ✅ Have its own `pyproject.toml`
- ✅ Use semantic versioning
- ✅ Include comprehensive README
- ✅ Have its own tests in `tests/`
- ✅ Follow Marie AI code standards
- ✅ Document compatibility with marie-ai versions

## Package Naming Convention

- Main package: `marie-ai` (contains core platform)
- Sub-packages: `marie-{name}` (e.g., `marie-mcp`, `marie-sdk`, `marie-cli`)
- First-party executable plugins: `marie-plugin-{name}`

Distributions that extend the public Marie Python API use the PEP 420 `marie`
namespace. Place their code under `src/marie/{name}/`, import it as
`marie.{name}`, and never add `src/marie/__init__.py`. Distribution names stay
hyphenated (`marie-agent`, `marie-instrumentation`) while Python imports use the
shared namespace (`marie.agent`, `marie.instrumentation`). Nested namespaces
are valid when a direct name is already owned; `marie-mcp` uses
`marie.mcp.server` and shares implicit `marie.mcp` with `marie-ai`. The `marie-ai`
serving facade is imported from `marie.runtime`, not from the namespace root.
`marie-cli` owns `marie.cli` and the console script; `marie-ai` owns the single
`marie/__main__.py` module launcher so existing source-file entry points remain
valid without placing a duplicate module in both wheels.

The root `pyproject.toml` declares each reusable local distribution as a uv path
source with `editable = true`. After `uv sync`, changes under any
`packages/*/src` namespace portion are immediately visible to Marie AI without
rebuilding a wheel. Release checks still build and install wheels in a clean
environment so editable source paths cannot hide packaging errors.

## Version Compatibility

Maintain compatibility matrix in each package README:

| marie-mcp | marie-ai | Status |
|-----------|----------|--------|
| 0.1.x     | 3.0.x    | ✅ Stable |

| marie-kernel | marie-ai | Status |
|--------------|----------|--------|
| 0.1.x        | 3.0.x    | 🚧 Development |

| marie-extension | marie-ai | Status |
|-----------------|----------|--------|
| 0.1.x           | 3.0.x    | 🚧 Development |

| marie-plugin-daemon | marie-ai | Status |
|---------------------|----------|--------|
| 0.1.x               | 3.0.x    | 🚧 Development |

| marie-plugin-document-extraction | marie-ai | Status |
|----------------------------------|----------|--------|
| 0.2.x                            | 3.0.x    | 🚧 Development |

## Future Packages

Potential packages to add:
- `marie-sdk` - Python SDK for application developers
- `marie-storage` - Storage adapters (S3, GCS, Azure)
- `marie-monitoring` - Observability tools

## Questions?

- Main docs: https://docs.marieai.co
- Issues: https://github.com/marieai/marie-ai/issues
- Discussions: https://github.com/marieai/marie-ai/discussions
