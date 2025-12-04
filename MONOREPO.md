# Marie AI Monorepo Guide

This repository uses a monorepo structure to manage multiple related packages.

## Structure

```
marie-ai/
├── marie/                          # Main Marie AI package (2-5GB)
│   ├── Core document processing
│   ├── ML models & executors
│   ├── Gateway & scheduler
│   └── Full platform capabilities
│
├── packages/                       # Lightweight packages
│   └── marie-mcp/                  # MCP server (~5MB)
│       ├── src/marie_mcp/
│       ├── tests/
│       ├── examples/
│       └── pyproject.toml
│
├── .github/workflows/              # Shared CI/CD
├── setup.py                        # Main package build
├── pyproject.toml                  # Root config + main package
└── MONOREPO.md                     # This file
```

## Why Monorepo?

### Benefits
✅ **Shared tooling** - One configuration for linting, formatting, testing
✅ **Coordinated releases** - Easy to keep packages in sync
✅ **Simplified development** - Clone once, work on everything
✅ **Atomic changes** - Update API contract across packages in one PR
✅ **Better testing** - Integration tests between packages
✅ **Unified CI/CD** - One pipeline for all packages

### Package Independence
✅ **Separate PyPI packages** - Install only what you need
✅ **Independent versioning** - Each package has its own version
✅ **Separate changelogs** - Clear package-specific history
✅ **Lightweight clients** - marie-mcp is only 5MB

## Installation

### For Users

**Install main Marie AI platform**:
```bash
pip install marie-ai
```

**Install MCP server** (lightweight, for AI assistants):
```bash
pip install marie-mcp
```

**Install both**:
```bash
pip install marie-ai marie-mcp
```

### For Developers

**Clone and install all packages**:
```bash
git clone https://github.com/marieai/marie-ai.git
cd marie-ai

# Install main package in editable mode
pip install -e .

# Install MCP package in editable mode
cd packages/marie-mcp
pip install -e .
cd ../..
```

**Or use a script**:
```bash
# Install all packages in dev mode
./scripts/install-all.sh
```

## Development Workflow

### Working on Main Package

```bash
# Make changes in marie/
vim marie/scheduler/job_scheduler.py

# Run tests
pytest tests/

# Run specific test
pytest tests/unit/scheduler/test_job_scheduler.py
```

### Working on MCP Package

```bash
# Make changes in packages/marie-mcp/
vim packages/marie-mcp/src/marie_mcp/tools/document_processing.py

# Run tests
cd packages/marie-mcp
pytest tests/

# Test locally
marie-mcp
```

### Working Across Packages

When making API changes that affect both packages:

```bash
# 1. Update main package API
vim marie/serve/runtimes/servers/marie_gateway.py

# 2. Update MCP client to match
vim packages/marie-mcp/src/marie_mcp/clients/marie_client.py

# 3. Run integration tests
pytest tests/integration/
cd packages/marie-mcp && pytest tests/integration/

# 4. Commit together
git add marie/ packages/marie-mcp/
git commit -m "feat: add new job status endpoint"
```

## Testing Strategy

### Unit Tests
Each package has its own unit tests:
```bash
# Main package
pytest tests/unit/

# MCP package
cd packages/marie-mcp && pytest tests/
```

### Integration Tests
Test interaction between packages:
```bash
# Tests that MCP client works with Marie gateway
pytest tests/integration/test_mcp_integration.py
```

### End-to-End Tests
Full workflow tests:
```bash
# Start Marie gateway
marie server --start --uses config/service/marie-dev.yml

# Run MCP e2e tests
cd packages/marie-mcp
pytest tests/e2e/
```

## Code Quality

### Shared Configuration

All packages use shared configuration in root `pyproject.toml`:
- **black** - Code formatting
- **isort** - Import sorting
- **mypy** - Type checking
- **flake8** - Linting (via setup.cfg)

### Running Checks

```bash
# Format all code
black marie/ packages/

# Sort imports
isort marie/ packages/

# Type check
mypy marie/ packages/

# Lint
flake8 marie/ packages/
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

Runs automatically on commit:
- black formatting
- isort import sorting
- trailing whitespace removal
- large file check

## CI/CD

### GitHub Actions Workflows

**Main Package CI** (`.github/workflows/ci-main.yml`):
- Tests on Python 3.10, 3.11, 3.12
- Full test suite including GPU tests
- Build Docker images
- Publish to PyPI on release

**MCP Package CI** (`.github/workflows/ci-mcp.yml`):
- Tests on Python 3.10, 3.11, 3.12
- Lightweight tests (no GPU needed)
- Integration tests against Marie gateway
- Publish to PyPI on release

**Integration Tests** (`.github/workflows/integration.yml`):
- Start Marie gateway in Docker
- Run MCP package against live gateway
- Verify API compatibility

### Running CI Locally

```bash
# Install act (GitHub Actions locally)
brew install act  # or apt-get install act

# Run main CI
act -W .github/workflows/ci-main.yml

# Run MCP CI
act -W .github/workflows/ci-mcp.yml
```

## Versioning

### Independent Versions

Each package has its own version:
- **marie-ai**: v3.0.30 (main package)
- **marie-mcp**: v0.1.0 (MCP package)

### Semantic Versioning

All packages follow [semver](https://semver.org/):
- **Major**: Breaking changes (e.g., 1.0.0 → 2.0.0)
- **Minor**: New features, backward compatible (e.g., 1.0.0 → 1.1.0)
- **Patch**: Bug fixes (e.g., 1.0.0 → 1.0.1)

### Version Compatibility

Document compatibility in each package README:

**packages/marie-mcp/README.md**:
```markdown
## Compatibility

| marie-mcp | marie-ai | Status |
|-----------|----------|--------|
| 0.1.x     | 3.0.x    | ✅ Stable |
| 0.2.x     | 3.1.x    | 🚧 Beta |
```

## Release Process

### Releasing Main Package

```bash
# 1. Update version in setup.py
vim setup.py  # version='3.0.31'

# 2. Update CHANGELOG.md
vim CHANGELOG.md

# 3. Commit and tag
git add setup.py CHANGELOG.md
git commit -m "chore: release v3.0.31"
git tag v3.0.31
git push origin main --tags

# 4. GitHub Actions will build and publish to PyPI
```

### Releasing MCP Package

```bash
# 1. Update version in pyproject.toml
cd packages/marie-mcp
vim pyproject.toml  # version = "0.1.1"

# 2. Update CHANGELOG.md
vim CHANGELOG.md

# 3. Commit and tag
git add pyproject.toml CHANGELOG.md
git commit -m "chore(marie-mcp): release v0.1.1"
git tag marie-mcp-v0.1.1
git push origin main --tags

# 4. GitHub Actions will build and publish to PyPI
```

### Coordinated Releases

When releasing breaking changes that affect both packages:

```bash
# 1. Make changes to both packages
git add marie/ packages/marie-mcp/

# 2. Update versions
vim setup.py  # marie-ai v4.0.0
vim packages/marie-mcp/pyproject.toml  # marie-mcp v1.0.0

# 3. Update changelogs
vim CHANGELOG.md
vim packages/marie-mcp/CHANGELOG.md

# 4. Commit with both tags
git commit -m "feat!: breaking API changes

BREAKING CHANGE: New job submission format

- marie-ai v4.0.0
- marie-mcp v1.0.0"

git tag v4.0.0
git tag marie-mcp-v1.0.0
git push origin main --tags
```

## Adding New Packages

To add a new package:

```bash
# 1. Create package directory
mkdir -p packages/your-package/src/your_package

# 2. Create pyproject.toml
cp packages/marie-mcp/pyproject.toml packages/your-package/
vim packages/your-package/pyproject.toml

# 3. Create package structure
cd packages/your-package
mkdir -p src/your_package tests examples
touch src/your_package/__init__.py
touch tests/__init__.py

# 4. Add README
vim README.md

# 5. Update root packages/README.md
vim ../../packages/README.md

# 6. Add CI workflow
cp ../../.github/workflows/ci-mcp.yml ../../.github/workflows/ci-your-package.yml
vim ../../.github/workflows/ci-your-package.yml
```

## Troubleshooting

### Import Issues

**Problem**: Can't import from other package

**Solution**: Install both packages in editable mode:
```bash
pip install -e .
cd packages/marie-mcp && pip install -e .
```

### Test Discovery Issues

**Problem**: pytest not finding tests

**Solution**: Run from package directory:
```bash
cd packages/marie-mcp
pytest tests/
```

### Version Conflicts

**Problem**: Conflicting dependencies between packages

**Solution**: Use compatible version ranges in `pyproject.toml`:
```toml
dependencies = [
    "httpx>=0.24.0,<1.0.0",
    "marie-ai>=3.0.0,<4.0.0"  # If you want to depend on main package
]
```

## Migration Notes

### From Separate Repo to Monorepo

**What changed**:
- marie-mcp moved from separate repo to `packages/marie-mcp/`
- Shared CI/CD in `.github/workflows/`
- Shared tooling configuration
- Coordinated release process

**What stayed the same**:
- Separate PyPI packages (`pip install marie-mcp`)
- Independent versioning (v0.1.0, v0.2.0, etc.)
- Separate changelogs
- Lightweight installation (still ~5MB)

**For users**: Nothing changed, install the same way
**For developers**: Clone one repo, work on both packages

## Questions?

- **General**: Open issue in main repo
- **MCP-specific**: Open issue with `[marie-mcp]` prefix
- **Architecture**: Discussions tab on GitHub
- **Support**: support@marieai.co

## Resources

- [Monorepo Best Practices](https://monorepo.tools/)
- [Python Monorepo Guide](https://www.tweag.io/blog/2023-04-04-python-monorepo-1/)
- [Semantic Versioning](https://semver.org/)
