# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Marie-AI is an AI-powered document processing framework. Neural networks for OCR, NER, classification, and extraction — exposed as agent-driven pipelines. Built with Python 3.10+, FastAPI, PyTorch, gRPC, and Celery.

## Essential Commands

```bash
# Code quality
make quality          # Check isort + flake8 (read-only)
make style            # Auto-fix isort
pre-commit run --all  # Run all hooks (detect-secrets, flake8, black, isort)

# Testing
make test                         # pytest -n auto --dist=loadfile -s -v ./tests/
pytest tests/ -k "test_name"      # Run specific test
pytest tests/unit/extract/ -x     # Run one test dir, stop on first failure

# Setup
pre-commit install    # Enable pre-commit hooks
```

## Architecture

### Runtime Stack

```
marie_server/__main__.py          # Server entry point
  └─ marie/serve/runtimes/
       ├── servers/marie_gateway.py   # FastAPI gateway (HTTP/gRPC ingress)
       ├── worker/request_handling.py # Worker process: dispatches to executors
       └── head/request_handling.py   # Head process: routes to worker replicas
```

Requests flow: **Client → Gateway → Head → Worker → Executor**. Workers run in separate processes (SPAWN), so executor `__init__` must re-initialize any process-local state (LLM tracking, CUDA context, etc.).

### Executors

The primary unit of document processing. Base class: `MarieExecutor` (`marie/executor/marie_executor.py`), which extends Jina's `Executor` with storage mixin, GPU health monitoring, and LLM tracking setup.

Key executor types:
- `marie/executor/extract/document_annotator_executor.py` — extraction orchestrator
- `marie/executor/classifier/` — document classification
- `marie/executor/ner/` — named entity recognition
- `marie/executor/rag/` — retrieval-augmented generation

Executors bind endpoints with `@requests(on="/endpoint")` and receive `DocList` + `parameters` dict.

### Extraction Pipeline

```
marie/extract/
├── annotators/          # LLM/ML annotators that produce raw output
│   ├── llm_annotator.py         # LLM-based extraction (supports refinement passes)
│   ├── llm_table_annotator.py   # Table-specific LLM extraction
│   ├── context_provider.py      # Injects context into prompts
│   └── util.py                  # scan_and_process_images, engine routing
├── results/core/
│   └── core_parsers.py          # Parses raw LLM JSON into StructuredRegions
├── engine/
│   ├── match_section_extract_visitor.py  # Maps regions → MatchSections with field extraction
│   └── transform.py             # Field value transforms (name parsing, formatting)
├── parser/
│   ├── base_region_parser.py    # Converts JSON → StructuredRegion with TableSeries
│   └── json_region_parser.py    # JSON-specific implementation
└── structures/
    ├── structured_region.py     # StructuredRegion, TableSeries, PageSpan
    └── unstructured_document.py # UnstructuredDocument (pages + lines + frames)
```

Data flow: **Annotator → core_parsers (JSON→Regions) → match_section_extract_visitor (Regions→Fields) → transform (field formatting)**

### Storage

`marie/storage/` provides `StorageManager` — a static facade routing `s3://`, `file://`, etc. to backend handlers. Key handler: `S3StorageHandler` in `s3_storage.py`. `StorageManager.read_to_file()` returns `True`/`False` to indicate success (callers should check for `False`).

### Configuration

Layout-specific configs live under `config/extract/TID-{layout_id}/`. Each layout has annotator configs (model, prompt path, processing mode) and field definitions (repeating/non-repeating with selectors, transforms, validators).

Config loading: `OmegaConf` + Hydra for YAML merging, `${VAR_NAME}` env interpolation.

## Development

- **Branch**: `develop-agents` (PR target: `main`)
- **Commits**: Conventional Commits required (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`)
- **Formatting**: black -S (single quotes), isort --profile black
- **Linting**: flake8 (max line 127, selects E9,F63,F7,F82)
- **Type hints**: Required on all function signatures. Use `from __future__ import annotations`.
- **Docstrings**: Google-style (Args/Returns/Raises) for public APIs only.
- **Imports**: Group as stdlib, third-party, local. isort handles ordering.
- **Testing**: pytest with asyncio_mode=auto, -n auto for parallel. Prefer real implementations over mocking.
- **Logging**: Always `MarieLogger("module.name")` — never `print()` or stdlib `logging`.
- **Exceptions**: Use hierarchy from `marie/excepts.py`. Never bare `except:`.

## Docker

Infrastructure compose files in `Dockerfiles/`:
- Core: `docker-compose.storage.yml` (PostgreSQL), `docker-compose.s3.yml` (MinIO), `docker-compose.rabbitmq.yml`, `docker-compose.etcd.yml`
- Observability: `docker-compose.clickhouse.yml`, `docker-compose.monitoring.yml`
- All-in-one: `docker-compose.allinone.yml` with profile-based tiers (`infra-only`, `observability`, `application`, `gpu`)

GPU images use CUDA 11.8 base. Network: `marie_default` (bridge).

## Agents and Skills

- Agent definitions: `.claude/agents/`
- Skills: `.claude/skills/`

See `AGENTS.md` for full style guide and pre-commit configuration.
