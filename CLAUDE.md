# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Marie-AI is an **Agentic Document Intelligence Platform** built in Python (>=3.10). It orchestrates autonomous AI agents for document processing tasks including OCR, classification, NER, and transformation. The system uses a DAG-driven execution model with advanced job scheduling and SLA support.

## Essential Commands

### Development Setup
```bash
# Install from source (editable mode)
pip install -e .

# Install with specific profile
MARIE_PIP_INSTALL_CORE=1 pip install .     # Minimal core
MARIE_PIP_INSTALL_PERF=1 pip install .     # Performance extras
pip install .[all]                          # All extras
```

### Testing
```bash
# Run all tests with parallel execution
make test
# Or: python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run specific test file
pytest tests/unit/serve/test_gateway.py

# Run specific test
pytest tests/unit/serve/test_gateway.py::TestGateway::test_start

# Run without slow tests
pytest -m "not slow" tests/

# Run with specific markers
pytest -m asyncio tests/
```

### Code Quality
```bash
# Check code quality (isort, flake8)
make quality

# Auto-format code (isort only; black is disabled)
make style

# Check/fix modified files only
make modified_only_fixup
```

### Running Marie Services
```bash
# Get help on CLI commands
marie -h

# Check version and environment
marie -vf

# Start server with config
marie server --start --uses config/service/marie.yml

# Start with development config
marie server --start --uses config/service/marie-dev.yml
```

### Docker Operations
```bash
# Build GPU image
DOCKER_BUILDKIT=1 docker build . \
  --build-arg PIP_TAG="standard" \
  -f ./Dockerfiles/gpu.Dockerfile \
  -t marieai/marie:3.0-cuda

# Run with custom entrypoint
docker run --rm -it --entrypoint /bin/bash marieai/marie:3.0.30-cuda
```

## Architecture Overview

### Core Package Structure

```
marie/
├── serve/                    # Runtime and gateway components
│   ├── executors/           # Base executor framework (metaclass registry, decorators)
│   └── runtimes/gateway/    # HTTP/gRPC/WebSocket gateways
├── orchestrate/             # Flow and deployment orchestration
│   ├── flow/               # DAG workflow builder
│   ├── deployments/        # Executor deployment manager
│   └── pods/               # Container/pod lifecycle
├── executor/               # Built-in document processing executors
├── job/                    # Job management and supervision
├── scheduler/              # Job scheduling engine with execution planners
├── state/                  # Distributed state management (StateStore, SlotCapacityManager)
├── core/                   # LLM-based agent framework
│   ├── agent/             # ReActAgent, AgentRunner, custom agents
│   ├── chat_engine/       # Conversational interfaces
│   └── storage/           # Vector stores and indexes
├── clients/               # gRPC/HTTP/WebSocket clients
├── models/                # Pre-trained ML models (CRAFT, ICR, Pix2Pix)
├── extract/               # Document extraction pipeline
├── storage/               # Storage backends (S3, PostgreSQL, local)
├── messaging/             # Message queue integration (RabbitMQ)
├── schemas/               # Configuration schemas
└── parsers/               # YAML/config parsers (JAML)
```

### Key Architectural Concepts

#### 1. Executor Framework
- **Metaclass-based registry**: All executors auto-register via `ExecutorType` metaclass
- **Decorator-based routing**: Use `@requests(on='/endpoint')` to define endpoints
- **Schema validation**: Methods wrapped with `_FunctionWithSchema` for automatic validation
- **Async-first**: Support for async/await with concurrent locking
- **Location**: `marie/serve/executors/__init__.py`

Example executor structure:
```python
class MyExecutor(Executor):
    @requests(on='/process')
    async def process(self, docs: DocumentArray, **kwargs):
        # Processing logic
        return docs
```

#### 2. Flow Orchestration (DAG Workflows)
- **Builder pattern**: Chain executors with `.add()` methods
- **Context manager support**: Use `with Flow() as f:` for lifecycle management
- **Multi-protocol**: Supports gRPC, HTTP, WebSocket
- **Location**: `marie/orchestrate/flow/base.py`

Example flow:
```python
f = (
    Flow(protocol='grpc')
    .add(uses=DocExtractor, name='extractor')
    .add(uses=Classifier, name='classifier')
    .add(uses=Indexer, name='indexer')
)

with f:
    f.start()
```

#### 3. Job Scheduling System
- **Execution planners**: HRRN, SJF, GlobalExecutionPlanner for DAG topology
- **SLA support**: Soft and hard SLA enforcement per job
- **State management**: `StateStore` for distributed state, `SlotCapacityManager` for resources
- **Job supervision**: `JobSupervisor` monitors execution and publishes events
- **Location**: `marie/scheduler/` and `marie/job/`

#### 4. Agent Framework
- **Multiple architectures**: ReActAgent, CustomSimpleAgent, FunctionCallingAgent, StructuredPlannerAgent
- **Task management**: `AgentRunner` handles create_task, run_step, get_task_output
- **Tool integration**: Agents can use external tools via function calling
- **Location**: `marie/core/agent/`

#### 5. Gateway System
- **Protocol translation**: HTTP ↔ gRPC conversion
- **Request routing**: Routes to appropriate executors
- **FastAPI-based**: HTTP gateway with OpenAPI docs, SSE, WebSocket support
- **Load balancing**: Distributes across shards/replicas
- **Location**: `marie/serve/runtimes/gateway/`

### Configuration System

Marie uses **YAML-first configuration** with dynamic variable substitution:

```yaml
jtype: Flow
version: '1'
protocol: grpc

shared_config:
  storage:
    psql:
      hostname: ${{ ENV.DB_HOSTNAME }}  # Environment variable substitution

toast:           # Event tracking
  native: { enabled: true }
  rabbitmq: { enabled: true }
```

Configuration files are in `config/service/`:
- `marie.yml` - Production configuration
- `marie-dev.yml` - Development configuration
- `deployment.yml` - Deployment templates

### Data Flow Patterns

**Typical document processing pipeline:**
```
Client → Gateway → Flow → [Executor Chain] → Gateway → Client
                           ├→ DocExtractor
                           ├→ Classifier
                           ├→ NERExtractor
                           └→ Indexer
```

**Agent execution flow:**
```
Task → AgentRunner.run_step() → Agent.chat() → [Reasoning Loop] → Result
                                  ├→ Think (LLM)
                                  ├→ Select tool
                                  ├→ Execute
                                  └→ Observe
```

**Job scheduling flow:**
```
Submit → JobScheduler → ExecutionPlanner → StateStore → JobSupervisor → Storage
```

### Testing Strategy

Tests are organized by scope:
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for component interactions
- `tests/docker_compose/` - Container-based integration tests

Key test fixtures in `tests/conftest.py`:
- `random_workspace_name` - Temporary workspace
- `port_generator` - Free port allocation
- `event_loop` - Async event loop

Tests use:
- `pytest` with `pytest-asyncio` for async tests
- `pytest-xdist` for parallel execution (`-n auto`)
- Markers: `@pytest.mark.slow`, `@pytest.mark.asyncio`, `@pytest.mark.timeout`

### Distributed Execution Features

- **Multi-shard/replica**: Horizontal scaling via shards and replicas
- **Consensus**: jraft module for distributed state consistency
- **Service discovery**: etcd-based discovery
- **Communication**: gRPC for inter-process communication
- **Observability**: OpenTelemetry instrumentation, Prometheus metrics, distributed tracing

### Storage Abstraction

Multiple storage backends with unified interface:
- **PostgreSQL**: Primary metadata and job state storage
- **S3**: Document and artifact storage
- **Local filesystem**: Development and testing
- **Message queues**: RabbitMQ for event streaming

Configuration via `shared_config.storage` in YAML files.

## Coding Standards

- **Formatter**: Code should be formatted with `isort` (black is currently disabled)
- **Linter**: `flake8` for linting
- **Pre-commit hooks**: Available via `.pre-commit-config.yaml`
- **Imports**: Use `isort` for consistent import ordering
- **Configuration**: `setup.cfg` and `pyproject.toml` contain tool configuration

## Naming Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- **Types**: feat, fix, docs, style, refactor, test, chore, perf, ci, build
- **Format**: `type: brief description in present tense`
- **Example**: `feat: add document classification executor`
- **Branch names**: Use lowercase, be descriptive, include type prefix

## Important Development Notes

### Python Version
- **Required**: Python >= 3.10 (enforced in `setup.py`)
- Check with: `python --version`

### CLI Entry Point
- Package name: `marie-ai` (on PyPI)
- Library name: `marie` (import name)
- CLI command: `marie` (defined in `setup.py` console_scripts)
- CLI implementation: `marie_cli/` package

### Environment Variables
Many configuration options support environment variable substitution using `${{ ENV.VARIABLE_NAME }}` syntax in YAML files.

### Async/Await Pattern
Marie is async-first throughout:
- Executors support both sync and async methods
- Use `asyncio=True` for async clients: `Client(asyncio=True)`
- Event loop management is automatic

### Workspace Management
Executors have workspace management for temporary file I/O. Workspaces are automatically created and cleaned up.

### Telemetry
- OpenTelemetry instrumentation throughout
- Can be disabled in tests via environment variables
- Prometheus metrics available via `@monitor` decorator

## Common Development Workflows

### Adding a New Executor
1. Create executor class inheriting from `Executor`
2. Use `@requests(on='/endpoint')` decorator for routing
3. Implement processing logic with `DocumentArray` parameters
4. Register automatically via metaclass
5. Add configuration YAML if needed
6. Write unit tests in `tests/unit/serve/executors/`

### Creating a New Flow
1. Define flow in YAML or programmatically
2. Chain executors with `.add(uses=ExecutorClass, name='name')`
3. Configure protocol (grpc/http/websocket)
4. Start with context manager or `.start()`
5. Test with integration tests in `tests/integration/`

### Debugging Tips
- Use `marie -vf` to check environment and version
- Enable debug logging in configuration
- Check gateway logs for request routing issues
- Use `pytest -s -v` for verbose test output with print statements
- Distributed tracing helps debug multi-executor flows

### Working with Jobs
- Submit jobs via `JobManager.submit_job()`
- Monitor state via `JobScheduler` API
- Query job status from `StateStore`
- Jobs are executed as DAGs with dependency management

## Resources

- Documentation: https://docs.marieai.co
- Issues: GitHub issue tracker
- PyPI: https://pypi.org/project/marie-ai/
- Docker Hub: marieai/marie
