---
sidebar_position: 2
---

# YAML configuration

Marie-AI configuration files support variables, substitution, and structured configuration. This guide covers YAML syntax, environment variables, and advanced configuration patterns.

## Variable substitution

Marie-AI YAML files support variable substitution using GitHub Actions-style syntax.

### Environment variables

Reference environment variables with `${{ ENV.VAR }}`:

```yaml
jtype: Flow
with:
  protocol: http
  port: ${{ ENV.MARIE_PORT }}

gateway:
  uses: MarieGateway
  with:
    job_scheduler_kwargs:
      hostname: ${{ ENV.POSTGRES_HOST }}
      port: ${{ ENV.POSTGRES_PORT }}
      username: ${{ ENV.POSTGRES_USER }}
      password: ${{ ENV.POSTGRES_PASSWORD }}
      database: ${{ ENV.POSTGRES_DB }}
```

Set environment variables before starting:

```bash
export MARIE_PORT=54322
export POSTGRES_HOST=localhost
export POSTGRES_PASSWORD=secret
marie server --uses config/marie.yml
```

### Default values

Provide fallback values for optional variables:

```yaml
gateway:
  with:
    job_scheduler_kwargs:
      hostname: ${{ ENV.POSTGRES_HOST | default: localhost }}
      port: ${{ ENV.POSTGRES_PORT | default: 5432 }}
      username: ${{ ENV.POSTGRES_USER | default: marie }}
      password: ${{ ENV.POSTGRES_PASSWORD }}
      database: ${{ ENV.POSTGRES_DB | default: marie }}
```

### Context variables

Pass variables at runtime with `${{ CONTEXT.VAR }}`:

```yaml
# flow.yml
jtype: Flow
with:
  port: ${{ CONTEXT.port }}

executors:
  - name: processor
    uses: ${{ CONTEXT.executor_class }}
    with:
      model_path: ${{ CONTEXT.model_path }}
```

Load with context in Python:

```python
from marie import Flow

context = {
    'port': 54322,
    'executor_class': 'MyExecutor',
    'model_path': '/models/latest'
}

f = Flow.load_config('flow.yml', context=context)
```

### Relative paths

Reference values within the same YAML file with `${{root.path.to.var}}`:

```yaml
# Note: no spaces around the path
shared:
  model_version: v2.0
  base_path: /opt/marie

executors:
  - name: ocr
    uses: OcrExecutor
    with:
      model_path: ${{root.shared.base_path}}/models/${{root.shared.model_version}}
```

:::tip
The difference between environment variables and relative paths is the presence of spaces:
- `${{ ENV.VAR }}` - environment variable (spaces)
- `${{root.path}}` - relative path (no spaces)
:::

## Flow configuration

### Basic structure

```yaml
jtype: Flow
version: '1'
with:
  # Protocol configuration
  protocol: [grpc, http]
  port: [54321, 54322]

  # Timeout settings
  timeout_send: 60000
  timeout_ready: 30000

  # Retry configuration
  retries: 3

gateway:
  uses: MarieGateway
  with:
    # Gateway-specific settings
    prefetch: 100

executors:
  - name: processor
    uses: ProcessorExecutor
```

### Protocol options

```yaml
with:
  # Single protocol
  protocol: http
  port: 54322

  # Multiple protocols
  protocol: [grpc, http, websocket]
  port: [54321, 54322, 54323]
```

| Protocol | Use case |
|----------|----------|
| `grpc` | High-performance, streaming |
| `http` | REST API, browser clients |
| `websocket` | Real-time, bidirectional |

### Timeout configuration

```yaml
with:
  # Request timeout (milliseconds)
  timeout_send: 60000

  # Startup timeout
  timeout_ready: 30000

  # Connection timeout
  timeout_ctrl: 10000
```

## Gateway configuration

### MarieGateway

```yaml
gateway:
  uses: MarieGateway
  with:
    # Database connection
    job_scheduler_kwargs:
      provider: postgresql
      hostname: ${{ ENV.POSTGRES_HOST | default: localhost }}
      port: ${{ ENV.POSTGRES_PORT | default: 5432 }}
      username: ${{ ENV.POSTGRES_USER | default: marie }}
      password: ${{ ENV.POSTGRES_PASSWORD }}
      database: ${{ ENV.POSTGRES_DB | default: marie }}

    # Key-value store
    kv_store_kwargs:
      provider: postgresql
      hostname: ${{ ENV.POSTGRES_HOST | default: localhost }}
      port: ${{ ENV.POSTGRES_PORT | default: 5432 }}
      username: ${{ ENV.POSTGRES_USER | default: marie }}
      password: ${{ ENV.POSTGRES_PASSWORD }}
      database: ${{ ENV.POSTGRES_DB | default: marie }}

    # Service discovery
    discovery_host: ${{ ENV.ETCD_HOST | default: localhost }}
    discovery_port: ${{ ENV.ETCD_PORT | default: 2379 }}
    discovery_service_name: marie

    # Rate limiting
    prefetch: 100
```

### Scheduler settings

```yaml
gateway:
  with:
    # Distributed scheduling
    distributed_scheduler: true

    # Worker pool
    max_workers: 10

    # Lease management
    lease_ttl_seconds: 5
    run_ttl_seconds: 120

    # DAG manager
    dag_manager:
      min_concurrent_dags: 2
      max_concurrent_dags: 32
      cache_ttl_seconds: 10
      dag_cache_size: 10000
      frontier_batch_size: 2000
```

## Executor configuration

### Basic executor

```yaml
executors:
  - name: processor
    uses: ProcessorExecutor
    with:
      model_path: /models/processor
      batch_size: 32
```

### Executor with replicas

```yaml
executors:
  - name: ocr
    uses: OcrExecutor
    replicas: 4
    with:
      device: cuda
```

### External executor (Docker)

```yaml
executors:
  - name: classifier
    uses: docker://marieai/classifier:latest
    with:
      model: document-classifier-v2
```

### Executor modules

```yaml
executors:
  - name: custom
    uses: CustomExecutor
    py_modules:
      - executors/custom.py
      - executors/utils.py
```

### Error handling

```yaml
executors:
  - name: processor
    uses: ProcessorExecutor
    exit_on_exceptions:
      - RuntimeError
      - OutOfMemoryError
      - torch.cuda.CUDAError
```

## Authentication configuration

```yaml
auth:
  keys:
    - name: backend-service
      api_key: ${{ ENV.API_KEY_BACKEND }}
      enabled: true

    - name: web-client
      api_key: ${{ ENV.API_KEY_WEB }}
      enabled: true
```

## Storage configuration

```yaml
storage:
  # Primary storage
  - provider: s3
    bucket: ${{ ENV.S3_BUCKET }}
    region: ${{ ENV.AWS_REGION | default: us-east-1 }}
    access_key: ${{ ENV.AWS_ACCESS_KEY_ID }}
    secret_key: ${{ ENV.AWS_SECRET_ACCESS_KEY }}

  # Fallback storage
  - provider: postgresql
    hostname: ${{ ENV.POSTGRES_HOST }}
    port: ${{ ENV.POSTGRES_PORT }}
    username: ${{ ENV.POSTGRES_USER }}
    password: ${{ ENV.POSTGRES_PASSWORD }}
    database: ${{ ENV.POSTGRES_DB }}
    table: documents
```

## Logging configuration

```yaml
logging:
  level: ${{ ENV.LOG_LEVEL | default: INFO }}
  format: json

  # Optional: file output
  file:
    path: /var/log/marie/marie.log
    max_size: 100MB
    max_files: 10
```

## LLM tracking configuration

Track LLM calls across your document processing pipelines for observability, cost management, and debugging.

### Basic configuration

```yaml
llm_tracking:
  enabled: true
  exporter: rabbitmq
  project_id: my-project
```

### Full configuration

```yaml
llm_tracking:
  enabled: true
  exporter: rabbitmq  # or "console" for development
  project_id: ${{ ENV.PROJECT_ID | default: marie-ai }}

  # Worker configuration
  worker:
    enabled: true  # Set to false if running worker separately

  # RabbitMQ configuration
  rabbitmq:
    hostname: ${{ ENV.RABBITMQ_HOST | default: localhost }}
    port: ${{ ENV.RABBITMQ_PORT | default: 5672 }}
    username: ${{ ENV.RABBITMQ_USER | default: guest }}
    password: ${{ ENV.RABBITMQ_PASSWORD }}
    vhost: ${{ ENV.RABBITMQ_VHOST | default: / }}
    exchange: llm-events
    queue: llm-ingestion
    routing_key: llm.event

  # PostgreSQL configuration (metadata storage)
  postgres:
    url: ${{ ENV.LLM_TRACKING_POSTGRES_URL }}
    # Or use individual fields:
    # hostname: ${{ ENV.POSTGRES_HOST }}
    # port: ${{ ENV.POSTGRES_PORT }}
    # username: ${{ ENV.POSTGRES_USER }}
    # password: ${{ ENV.POSTGRES_PASSWORD }}
    # database: marie

  # S3 configuration (payload storage)
  s3:
    bucket: ${{ ENV.LLM_TRACKING_S3_BUCKET | default: marie-llm-tracking }}

  # ClickHouse configuration (analytics)
  clickhouse:
    host: ${{ ENV.CLICKHOUSE_HOST | default: localhost }}
    port: ${{ ENV.CLICKHOUSE_HTTP_PORT | default: 8123 }}
    native_port: ${{ ENV.CLICKHOUSE_NATIVE_PORT | default: 9000 }}
    database: ${{ ENV.CLICKHOUSE_DB | default: marie_llm }}
    user: ${{ ENV.CLICKHOUSE_USER | default: default }}
    password: ${{ ENV.CLICKHOUSE_PASSWORD }}
    batch_size: 1000
    flush_interval_s: 5.0
```

### Exporter options

| Exporter | Use case |
|----------|----------|
| `console` | Development and debugging |
| `rabbitmq` | Production with full durability |

:::warning
The `rabbitmq` exporter requires both PostgreSQL and S3 to be configured. The tracker will fail to start if storage is not properly configured.
:::

### Minimal production setup

```yaml
llm_tracking:
  enabled: true
  exporter: rabbitmq
  project_id: production

  rabbitmq:
    hostname: ${{ ENV.RABBITMQ_HOST }}
    port: 5672
    username: ${{ ENV.RABBITMQ_USER }}
    password: ${{ ENV.RABBITMQ_PASSWORD }}
    exchange: llm-events
    queue: llm-ingestion
    routing_key: llm.event

  postgres:
    url: ${{ ENV.LLM_TRACKING_POSTGRES_URL }}

  s3:
    bucket: ${{ ENV.LLM_TRACKING_S3_BUCKET }}

  clickhouse:
    host: ${{ ENV.CLICKHOUSE_HOST }}
    database: marie_llm
```

For detailed usage, see the [LLM tracking guide](../../guides/llm-tracking.md).

## Complete example

```yaml
jtype: Flow
version: '1'
with:
  protocol: [grpc, http]
  port: [54321, 54322]
  timeout_send: 120000
  retries: 3

gateway:
  uses: MarieGateway
  with:
    job_scheduler_kwargs:
      provider: postgresql
      hostname: ${{ ENV.POSTGRES_HOST | default: localhost }}
      port: ${{ ENV.POSTGRES_PORT | default: 5432 }}
      username: ${{ ENV.POSTGRES_USER | default: marie }}
      password: ${{ ENV.POSTGRES_PASSWORD }}
      database: ${{ ENV.POSTGRES_DB | default: marie }}

    kv_store_kwargs:
      provider: postgresql
      hostname: ${{ ENV.POSTGRES_HOST | default: localhost }}
      port: ${{ ENV.POSTGRES_PORT | default: 5432 }}
      username: ${{ ENV.POSTGRES_USER | default: marie }}
      password: ${{ ENV.POSTGRES_PASSWORD }}
      database: ${{ ENV.POSTGRES_DB | default: marie }}

    discovery_host: ${{ ENV.ETCD_HOST | default: localhost }}
    discovery_port: ${{ ENV.ETCD_PORT | default: 2379 }}
    discovery_service_name: marie

    distributed_scheduler: true
    max_workers: 10

    dag_manager:
      max_concurrent_dags: 32
      frontier_batch_size: 2000

    prefetch: 100

executors:
  - name: ocr
    uses: OcrExecutor
    replicas: 2
    with:
      device: ${{ ENV.DEVICE | default: cuda }}
      model_path: /models/ocr
    exit_on_exceptions:
      - torch.cuda.CUDAError

  - name: extract
    uses: ExtractExecutor
    replicas: 4
    with:
      model_path: /models/extract

auth:
  keys:
    - name: production
      api_key: ${{ ENV.API_KEY }}
      enabled: true

logging:
  level: ${{ ENV.LOG_LEVEL | default: INFO }}
  format: json

llm_tracking:
  enabled: true
  exporter: rabbitmq
  project_id: production

  rabbitmq:
    hostname: ${{ ENV.RABBITMQ_HOST | default: localhost }}
    port: 5672
    username: ${{ ENV.RABBITMQ_USER | default: guest }}
    password: ${{ ENV.RABBITMQ_PASSWORD }}
    exchange: llm-events
    queue: llm-ingestion
    routing_key: llm.event

  postgres:
    url: ${{ ENV.LLM_TRACKING_POSTGRES_URL }}

  s3:
    bucket: ${{ ENV.LLM_TRACKING_S3_BUCKET }}

  clickhouse:
    host: ${{ ENV.CLICKHOUSE_HOST | default: localhost }}
    database: marie_llm
```

## Validation

### Check configuration syntax

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/marie.yml'))"
```

### Test variable substitution

```python
from marie import Flow

# Load with dry_run to check configuration
f = Flow.load_config('config/marie.yml')
print(f.args)  # Inspect resolved configuration
```

## Best practices

### 1. Use environment variables for secrets

```yaml
# Good: secrets from environment
password: ${{ ENV.DB_PASSWORD }}

# Bad: hardcoded secrets
password: my-secret-password
```

### 2. Provide defaults for optional settings

```yaml
# Good: sensible defaults
port: ${{ ENV.PORT | default: 54322 }}

# Risky: may fail if not set
port: ${{ ENV.PORT }}
```

### 3. Group related configuration

```yaml
# Good: organized sections
database:
  host: ${{ ENV.DB_HOST }}
  port: ${{ ENV.DB_PORT }}
  name: ${{ ENV.DB_NAME }}
```

### 4. Document required variables

```yaml
# Required environment variables:
# - POSTGRES_PASSWORD: Database password
# - API_KEY: Authentication key
#
# Optional (with defaults):
# - POSTGRES_HOST: Database host (default: localhost)
# - LOG_LEVEL: Logging level (default: INFO)
```

### 5. Use separate configs per environment

```text
config/
├── marie.yml           # Base configuration
├── marie.dev.yml       # Development overrides
├── marie.staging.yml   # Staging overrides
└── marie.prod.yml      # Production overrides
```

## Next steps

- [Basic config](./config.md) - Service configuration overview
- [LLM tracking guide](../../guides/llm-tracking.md) - Track LLM calls for observability
- [Error handling](../../guides/error-handling.md) - Exception handling
- [Gateway guide](../../guides/gateway.md) - Gateway configuration details
