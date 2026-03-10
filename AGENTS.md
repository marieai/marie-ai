# AGENTS.md

Marie-AI is an AI-powered document processing framework built on Python. Neural networks for OCR, NER, classification, and extraction — exposed as agent-driven pipelines.

## Branches

- Default development branch: `develop-agents`
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`)

## Commands

```bash
make quality          # Check isort + flake8 (read-only)
make style            # Auto-fix isort
make test             # pytest -n auto --dist=loadfile -s -v ./tests/
pytest tests/ -k "test_name"   # Run specific test
pre-commit install    # Enable hooks (detect-secrets, flake8, black, isort)
pre-commit run --all  # Run all hooks
```

## Style Guide

### Formatting

- **black** with `-S` (skip string normalization — use single quotes)
- **isort** with `--profile black`
- **flake8**: max line length 127, selects `E9,F63,F7,F82`
- Run `pre-commit run --all` before pushing

### Naming

- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Private attributes: `_single_leading_underscore`

### Type Hints

- Always use type hints on function signatures
- Use `from __future__ import annotations` for forward references
- Use `TYPE_CHECKING` guard for import-only types to avoid circular imports:
  ```python
  from __future__ import annotations
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      from marie.types import Document
  ```
- Prefer `X | None` over `Optional[X]` in new code

### Docstrings

Google-style with Args/Returns/Raises:

```python
def process(doc: Document, threshold: float = 0.5) -> Result:
    """Process a document through the extraction pipeline.

    Args:
        doc: The input document.
        threshold: Confidence threshold for extraction.

    Returns:
        Extraction result with confidence scores.

    Raises:
        ProcessingError: If extraction fails.
    """
```

### Imports

Group as: stdlib, third-party, local. isort handles ordering.

```python
import os
from pathlib import Path

import torch
from pydantic import BaseModel

from marie.excepts import ProcessingError
from marie.logging_core.logger import MarieLogger
```

### Error Handling

- Use custom exceptions from `marie/excepts.py` — `BaseMarieException` hierarchy
- Key exceptions: `ProcessingError`, `ExecutorError`, `BadConfigSource`, `MaxTokensExceededError`, `RepetitionError`
- Never use bare `except:` — always catch specific exceptions
- Use `raise ... from e` to chain exceptions

### Agent Framework Patterns

- Executors use `@requests` decorator for endpoint binding
- Agent configs are Pydantic `BaseModel` subclasses in `marie/agent/config.py`
- Skills follow the `agentskills.io` spec — see `marie/agent/skills/models.py`
- Use `MarieLogger("module.name")` for logging, not `print()` or stdlib `logging`

### Configuration

- Use Pydantic `BaseModel` with `Field(...)` for config classes
- Support YAML-based configuration loading
- Environment variable interpolation with `${VAR_NAME}` syntax

### Avoid

- Mutable default arguments (`def f(items=[])` — use `None` + assign)
- Star imports (`from module import *`)
- `object` without proper typing as a catch-all
- `print()` for logging — use `MarieLogger`
- Bare `except:` or `except Exception:`

## Testing

- pytest with `asyncio_mode = auto` (no need for `@pytest.mark.asyncio`)
- Use `-n auto` for parallel test execution
- Prefer testing real implementations over mocking
- Test markers: `@pytest.mark.slow`, `@pytest.mark.timeout`
- Test files go in `tests/` mirroring the `marie/` structure

## Pre-commit

Hooks configured in `.pre-commit-config.yaml`:
1. `detect-secrets` — prevents committing secrets
2. `no-commit-to-branch` — blocks direct commits to `main`/`master`
3. `flake8` — linting
4. `black -S` — formatting
5. `isort --profile black` — import sorting

Run `pre-commit install` after cloning.
