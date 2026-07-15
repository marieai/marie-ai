# Marie AI Packages

This directory contains separately installable packages that are part of the Marie AI ecosystem.

## Available Packages

### 📦 marie-mcp
**Lightweight MCP server for AI assistant integration**

A Model Context Protocol (MCP) server that enables AI assistants like Claude to interact with Marie AI's document intelligence capabilities.

- **Size**: ~5MB (vs 2-5GB for main marie-ai package)
- **Purpose**: Client-side integration for AI assistants
- **Install**: `uv add marie-mcp`
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
- **Docs**: [packages/marie-extension/README.md](./marie-extension/README.md)

**Features**:
- YAML-first extension package schema
- Standard ZIP discovery by `marie-extension.yaml`
- Safe path validation for package files
- Deny-by-default permission models

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
| Manifest contract | `marie-extension` | `marie_extension` | Marie control plane, build/CI tooling | Validates `marie-extension.yaml` packages (schema, ZIP loading, permissions). Metadata only — never executes plugin code |
| Plugin host | `marie-plugin-daemon` | n/a (Go) | Daemon process | Installs, starts, and invokes plugins; creates their uv environments; embeds and injects the Python runtime |
| Plugin-side runtime | `marie-plugin-runtime` (source lives in `marie-plugin-daemon/python_runtime/`) | `marie_plugins.runtime` | Inside each plugin process | Stdio protocol shim: frames, sessions, heartbeat, test client. Stdlib-only. Daemon-provided in production; dev-only wheel for plugin authors |
| Plugins | `marie-plugin-{name}` | `marie_plugins.{name}` | Inside their own plugin process | Actual plugin logic (e.g. `marie_plugins.document_extraction`) |

Import rules that keep the boundary honest:

- Plugin code imports only `marie_plugins.runtime` and its own locked dependencies — never `marie` or `marie_extension`.
- Host code never imports plugin internals; it validates manifests with `marie_extension` and talks to plugins through the daemon.
- `marie_plugins` is a PEP 420 implicit namespace shared by the runtime and every plugin. Never create `marie_plugins/__init__.py` — a regular package at that name shadows the other half of the namespace.

Disambiguation: `marie_extension/runtime.py` (the manifest's runtime *envelope model* — network policy, resource limits) is unrelated to `marie_plugins.runtime` (the in-process protocol library).

## Monorepo Structure

```
marie-ai/
├── marie/                      # Main Marie AI package (server-side)
│   ├── Core processing
│   ├── ML models & executors
│   └── Gateway & scheduler
│
└── packages/                   # Additional packages (client-side)
    ├── marie-mcp/              # MCP server for AI assistants
    │   ├── src/marie_mcp/
    │   ├── pyproject.toml      # Separate PyPI package
    │   └── README.md
    ├── marie-kernel/           # State management kernel
    │   ├── src/marie_kernel/
    │   ├── pyproject.toml      # Separate PyPI package
    │   └── README.md
    ├── marie-extension/        # Extension package schema and validator
    │   ├── src/marie_extension/
    │   ├── pyproject.toml      # Separate PyPI package
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
3. Create `src/your_package/` structure
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
- `marie-cli` - Enhanced CLI tools
- `marie-storage` - Storage adapters (S3, GCS, Azure)
- `marie-monitoring` - Observability tools

## Questions?

- Main docs: https://docs.marieai.co
- Issues: https://github.com/marieai/marie-ai/issues
- Discussions: https://github.com/marieai/marie-ai/discussions
