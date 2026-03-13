# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Marie-AI is an AI-powered document processing framework. Neural networks for OCR, NER, classification, and extraction — exposed as agent-driven pipelines. Built with Python, FastAPI, PyTorch, and Celery.

## Essential Commands

```bash
# Code quality
make quality          # Check isort + flake8 (read-only)
make style            # Auto-fix isort
pre-commit run --all  # Run all hooks (detect-secrets, flake8, black, isort)

# Testing
make test             # pytest -n auto --dist=loadfile -s -v ./tests/
pytest tests/ -k "test_name"   # Run specific test

# Setup
pre-commit install    # Enable pre-commit hooks
```

## Architecture

### Key Directories

```
marie/
├── executor/          # Document processing executors (@requests decorator)
├── api/               # FastAPI REST endpoints
├── agent/             # Agent framework — configs, skills, orchestration
│   ├── config.py      # Pydantic BaseModel agent configs
│   └── skills/        # agentskills.io spec implementations
├── logging_core/      # MarieLogger — use instead of print() or stdlib logging
├── excepts.py         # Custom exception hierarchy (BaseMarieException)
├── models/            # ML model definitions and loaders
├── components/        # Reusable pipeline components
└── utils/             # Shared utilities
tests/                 # Mirrors marie/ structure
```

### Core Patterns

**Executors**: Use `@requests` decorator for endpoint binding. Executors are the primary unit of document processing.

**Agent Framework**: Configs are Pydantic `BaseModel` subclasses. Skills follow `agentskills.io` spec.

**Logging**: Always use `MarieLogger("module.name")` — never `print()` or stdlib `logging`.

**Configuration**: Pydantic `BaseModel` with `Field(...)`. YAML-based loading. Environment variable interpolation with `${VAR_NAME}`.

**Exceptions**: Use custom exceptions from `marie/excepts.py` — `ProcessingError`, `ExecutorError`, `BadConfigSource`, `MaxTokensExceededError`, `RepetitionError`.

## Development

- **Branch**: `develop-agents`
- **Formatting**: black -S (single quotes), isort --profile black
- **Linting**: flake8 (max line 127, selects E9,F63,F7,F82)
- **Type hints**: Required on all function signatures. Use `from __future__ import annotations`.
- **Docstrings**: Google-style (Args/Returns/Raises) for public APIs only.
- **Imports**: Group as stdlib, third-party, local. isort handles ordering.
- **Testing**: pytest with asyncio_mode=auto, -n auto for parallel. Prefer real implementations over mocking.

## Agents and Skills

- Agent definitions: `.claude/agents/`
- Skills: `.claude/skills/`

See `AGENTS.md` for full style guide and pre-commit configuration.
