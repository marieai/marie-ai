# Marie AI Packages

This directory contains lightweight packages that are part of the Marie AI monorepo.

## Available Packages

### marie-mcp

MCP (Model Context Protocol) server for the Marie AI document intelligence platform.

- **Location**: `packages/marie-mcp/`
- **PyPI**: [`marie-mcp`](https://pypi.org/project/marie-mcp/)
- **Size**: ~5MB (lightweight, no ML models)
- **Purpose**: AI assistant integration for document processing
- **Documentation**: [packages/marie-mcp/README.md](marie-mcp/README.md)

**Installation**:
```bash
pip install marie-mcp
```

**Development**:
```bash
cd packages/marie-mcp
pip install -e .
```

## Package Structure

Each package follows this structure:
```
packages/your-package/
├── src/
│   └── your_package/
│       ├── __init__.py
│       └── ...
├── tests/
│   ├── __init__.py
│   └── test_*.py
├── examples/
│   └── *.py
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

## Development

### Installing All Packages

From the root of the monorepo:
```bash
# Install main package
pip install -e .

# Install all packages
for pkg in packages/*/; do
    (cd "$pkg" && pip install -e .)
done
```

### Running Tests

```bash
# Test specific package
cd packages/marie-mcp
pytest tests/

# Test all packages from root
pytest packages/*/tests/
```

### Code Quality

All packages share the same code quality tools configured in the root `pyproject.toml`:
```bash
# Format code
black packages/
isort packages/

# Type check
mypy packages/

# Lint
flake8 packages/
```

## Adding a New Package

See [MONOREPO.md](../MONOREPO.md#adding-new-packages) for instructions on adding new packages.

## Versioning

Each package has independent versioning:
- Packages follow [Semantic Versioning](https://semver.org/)
- Version numbers are managed in each package's `pyproject.toml`
- Releases are tagged with `<package-name>-v<version>` (e.g., `marie-mcp-v0.1.0`)

## Publishing

Packages are published independently to PyPI:
```bash
# Build package
cd packages/marie-mcp
python -m build

# Publish (done automatically via GitHub Actions)
python -m twine upload dist/*
```

## Resources

- [Monorepo Guide](../MONOREPO.md)
- [Main Package Documentation](../README.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
