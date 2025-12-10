# Marie AI Packages

This directory contains separately installable packages that are part of the Marie AI ecosystem.

## Available Packages

### 📦 marie-mcp
**Lightweight MCP server for AI assistant integration**

A Model Context Protocol (MCP) server that enables AI assistants like Claude to interact with Marie AI's document intelligence capabilities.

- **Size**: ~5MB (vs 2-5GB for main marie-ai package)
- **Purpose**: Client-side integration for AI assistants
- **Install**: `pip install marie-mcp`
- **Docs**: [packages/marie-mcp/README.md](./marie-mcp/README.md)

**Use cases**:
- Claude Desktop integration
- LangChain agents
- OpenAI Agents SDK
- Custom AI assistant integrations

## Monorepo Structure

```
marie-ai/
├── marie/                      # Main Marie AI package (server-side)
│   ├── Core processing
│   ├── ML models & executors
│   └── Gateway & scheduler
│
└── packages/                   # Additional packages (client-side)
    └── marie-mcp/              # MCP server for AI assistants
        ├── src/marie_mcp/
        ├── pyproject.toml      # Separate PyPI package
        └── README.md
```

## Development

### Installing from Source

```bash
# Install main Marie AI package
pip install -e .

# Install MCP package
cd packages/marie-mcp
pip install -e .
```

### Publishing

Each package is published independently to PyPI:

```bash
# Publish main package
python -m build
twine upload dist/*

# Publish MCP package
cd packages/marie-mcp
python -m build
twine upload dist/*
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

## Version Compatibility

Maintain compatibility matrix in each package README:

| marie-mcp | marie-ai | Status |
|-----------|----------|--------|
| 0.1.x     | 3.0.x    | ✅ Stable |

## Future Packages

Potential packages to add:
- `marie-sdk` - Python SDK for application developers
- `marie-cli` - Enhanced CLI tools
- `marie-storage` - Storage adapters (S3, GCS, Azure)
- `marie-monitoring` - Observability tools
- `marie-plugins` - Plugin system

## Questions?

- Main docs: https://docs.marieai.co
- Issues: https://github.com/marieai/marie-ai/issues
- Discussions: https://github.com/marieai/marie-ai/discussions
