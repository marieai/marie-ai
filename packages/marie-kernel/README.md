# Marie State Kernel

State management kernel for Marie AI DAG task execution. Enables tasks within a DAG run to share state via simple key-value operations.

## Installation

```bash
# Core package (in-memory backend only)
pip install marie-kernel

# With PostgreSQL support
pip install marie-kernel[postgres]

# With Amazon S3 support
pip install marie-kernel[s3]

# All optional dependencies
pip install marie-kernel[all]
```

## Quick Start

```python
from marie_kernel import TaskInstanceRef, RunContext, create_backend

# Create a backend (in-memory for testing)
backend = create_backend("memory")

# Create task instance reference
ti = TaskInstanceRef(
    tenant_id="acme_corp",
    dag_name="document_pipeline",
    dag_id="run_2024_001",
    task_id="extract_text",
    try_number=1,
)

# Create context for the task
ctx = RunContext(ti, backend)

# Store state
ctx.set("EXTRACTED_TEXT", "Hello World")
ctx.set("TABLE_DATA", {"rows": [[1, 2], [3, 4]], "columns": ["a", "b"]})

# Retrieve state
text = ctx.get("EXTRACTED_TEXT")
tables = ctx.get("TABLE_DATA")
```

## Usage Patterns

### Cross-Task State Passing

Tasks can read state from upstream tasks in the same DAG run:

```python
# In upstream OCR task
def ocr_task(ctx: RunContext):
    lines = perform_ocr(document)
    ctx.set("OCR_LINES", lines)


# In downstream processing task
def process_task(ctx: RunContext):
    # Read from upstream task by task_id
    ocr_lines = ctx.get("OCR_LINES", from_task="ocr")

    # Process and store result
    tables = locate_tables(ocr_lines)
    ctx.set("TABLE_STRUCTS", tables)
```

### Fan-In Pattern (Multiple Upstream Tasks)

```python
def aggregator_task(ctx: RunContext):
    # Pull from multiple upstream tasks
    result = ctx.pull(
        "PARTIAL_RESULT", from_tasks=["processor_1", "processor_2", "processor_3"]
    )
    # Returns first match found in task order
```

### Default Values

```python
# Returns None if key doesn't exist
value = ctx.get("OPTIONAL_KEY")

# Returns specified default
config = ctx.get("CONFIG", default={"retries": 3})
```

## API Reference

### TaskInstanceRef

Immutable dataclass identifying a task execution attempt:

```python
@dataclass(frozen=True)
class TaskInstanceRef:
    tenant_id: str  # Multi-tenant isolation
    dag_name: str  # DAG name/type (e.g., "document_processing")
    dag_id: str  # Unique run identifier for this DAG execution
    task_id: str  # Task identifier within DAG
    try_number: int  # Retry attempt (1-indexed)
```

Factory methods:
- `TaskInstanceRef.from_dict(data, tenant_id="default")` - Create from dictionary
- `TaskInstanceRef.from_work_info(work_info, tenant_id="default")` - Create from WorkInfo object
- `ti.with_try_number(n)` - Create new ref with updated try_number

### RunContext

Primary API for task state operations:

```text
class RunContext:
    # Primary API (simple key-value)
    def set(self, key: str, value: Any) -> None
    def get(self, key: str, *, from_task: str = None, default: Any = None) -> Any

    # Advanced API (with metadata, multi-task pulls)
    def push(self, key: str, value: Any, *, metadata: dict = None) -> None
    def pull(self, key: str, *, from_task: str = None, from_tasks: list = None, default: Any = None) -> Any

    # Properties
    @property
    def ti(self) -> TaskInstanceRef
```

### Backends

#### InMemoryStateBackend

Thread-safe in-memory backend for testing:

```python
from marie_kernel.backends import InMemoryStateBackend

backend = InMemoryStateBackend()
```

#### PostgresStateBackend

Production backend using PostgreSQL:

```python
from psycopg_pool import ConnectionPool
from marie_kernel.backends import PostgresStateBackend

pool = ConnectionPool("postgresql://user:pass@localhost/marie")
backend = PostgresStateBackend(pool)
```

#### S3StateBackend

Distributed backend using Amazon S3 (suitable for serverless/distributed environments):

```python
import boto3
from marie_kernel.backends import S3StateBackend

s3_client = boto3.client("s3")
backend = S3StateBackend(s3_client, bucket="my-state-bucket", prefix="marie-state")
```

Object keys follow the pattern:
```
{prefix}/{tenant_id}/{dag_name}/{dag_id}/{task_id}/{try_number}/{key}.json
```

**Note**: S3 provides eventual consistency. Use PostgreSQL if strong consistency is required.

### Factory Functions

```python
from marie_kernel import create_backend, create_backend_from_url

# By type
backend = create_backend("memory")
backend = create_backend("postgres", connection_pool=pool)
backend = create_backend("s3", s3_client=s3_client, bucket="my-bucket")

# By URL
backend = create_backend_from_url("memory://")
backend = create_backend_from_url("postgresql://user:pass@localhost/marie")
backend = create_backend_from_url("s3://my-bucket/marie-state")
```

## Database Schema

For PostgreSQL, create the required table:

```sql
-- See migrations/001_task_state.sql
CREATE TABLE task_state (
    tenant_id     TEXT NOT NULL,
    dag_name      TEXT NOT NULL,
    dag_id        TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    try_number    INT NOT NULL,
    key           TEXT NOT NULL,
    value_json    JSONB NOT NULL,
    metadata      JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, dag_name, dag_id, task_id, try_number, key)
);
```

## Retry Semantics

When retrying a task:

1. **Clear state BEFORE retry**: Call `backend.clear_for_task(ti)` before incrementing `try_number`
2. **Increment try_number**: Update the task's try_number in scheduler metadata
3. **Execute retry**: New attempt starts with clean slate

```python
# In scheduler retry logic
async def schedule_retry(job_id, state_backend):
    job = await get_job(job_id)

    # Step 1: Clear stale state
    ti = TaskInstanceRef.from_work_info(job)
    state_backend.clear_for_task(ti)

    # Step 2: Increment try_number
    new_try = job.data.get("try_number", 1) + 1
    await update_job(job_id, try_number=new_try)
```

## Multi-Tenant Isolation

State is isolated by `tenant_id` in the primary key. Different tenants cannot access each other's state:

```python
# Tenant A's state
ti_a = TaskInstanceRef(tenant_id="tenant_a", dag_id="dag", ...)
ctx_a = RunContext(ti_a, backend)
ctx_a.set("SECRET", "tenant_a_data")

# Tenant B cannot access it
ti_b = TaskInstanceRef(tenant_id="tenant_b", dag_id="dag", ...)
ctx_b = RunContext(ti_b, backend)
ctx_b.get("SECRET")  # Returns None
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/marie_kernel

# Formatting
black src tests
isort src tests
```

## License

MIT License - see LICENSE file for details.
