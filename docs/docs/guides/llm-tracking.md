---
sidebar_position: 12
---

# LLM tracking and observability

Marie-AI includes a comprehensive LLM tracking system for observability, cost management, and debugging of LLM-powered workflows. This system captures traces, observations, and metrics from LLM calls and stores them for analytics.

## Architecture overview

The tracking system follows a durable, async-first architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LLM Tracking Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Tracker                                                                    │
│   ┌─────┐                                                                    │
│   │Event│──┬─► S3 (payload)                                                 │
│   └─────┘  │                                                                 │
│            ├─► PostgreSQL (metadata + status='pending')                     │
│            │                                                                 │
│            └─► RabbitMQ (notification: event_id only)                       │
│                    │                                                         │
│                    ▼                                                         │
│   Worker ◄────────────────────────────────────────────────────              │
│   ┌─────────────┐                                                           │
│   │ Consume     │─► Lookup metadata in Postgres (by event_id)               │
│   │ Message     │─► Fetch payload from S3 (by s3_key)                       │
│   │             │─► Process & write to ClickHouse                           │
│   │             │─► Mark status='processed' in Postgres                     │
│   │             │─► ACK message in RabbitMQ                                 │
│   └─────────────┘                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| **Tracker** | Captures LLM events (traces, observations, scores) |
| **S3** | Stores all payload data (prompts, responses) |
| **PostgreSQL** | Stores metadata for analytics and processing status |
| **RabbitMQ** | Async event queue for worker processing |
| **Worker** | Consumes events, normalizes data, writes to ClickHouse |
| **ClickHouse** | Analytics database for dashboards and queries |

### Data flow

1. **Tracker** receives an LLM event (generation start, end, error)
2. Payload is saved to **S3** (prompts, responses, raw LLM data)
3. Metadata is saved to **PostgreSQL** (tokens, cost, latency, model info)
4. Lightweight notification (event_id only) is published to **RabbitMQ**
5. **Worker** consumes the message, fetches data, processes it
6. Normalized data is written to **ClickHouse** for analytics
7. Event is marked as processed in PostgreSQL

## Configuration

Configure LLM tracking in your YAML config file:

```yaml
llm_tracking:
  enabled: true
  exporter: rabbitmq  # or "console" for development
  project_id: my-project

  # Worker configuration
  worker:
    enabled: true  # Set to false if running worker separately

  # RabbitMQ configuration
  rabbitmq:
    hostname: localhost
    port: 5672
    username: guest
    password: guest
    virtualhost: /  # Optional: specify vhost
    exchange: llm-events
    queue: llm-ingestion
    routing_key: llm.event

  # PostgreSQL configuration (metadata storage)
  postgres:
    url: postgresql://user:password@localhost:5432/marie
    # Or use individual fields:
    # hostname: localhost
    # port: 5432
    # username: postgres
    # password: secret
    # database: marie

  # S3 configuration (payload storage)
  # Uses shared storage.s3 config if not specified here
  s3:
    bucket: marie-llm-tracking

  # ClickHouse configuration (analytics)
  clickhouse:
    host: localhost
    port: 8123
    native_port: 9000
    database: marie
    user: default
    password: ""
    batch_size: 1000
    flush_interval_s: 5.0

# Shared S3 storage (used by llm_tracking if s3.bucket not specified)
storage:
  s3:
    enabled: true
    endpoint: http://localhost:9000
    access_key_id: minioadmin
    secret_access_key: minioadmin
    bucket_name: marie
```

## Usage

### Basic tracking

```python
from marie.llm_tracking import get_tracker

# Get the singleton tracker instance
tracker = get_tracker()

# Create a trace for a user request
trace_id = tracker.create_trace(
    name="document-processing",
    user_id="user-123",
    session_id="session-456",
    metadata={"document_type": "invoice"},
)

# Start tracking an LLM generation
observation_id = tracker.generation(
    trace_id=trace_id,
    name="extract-fields",
    model="gpt-4",
    input={"prompt": "Extract invoice fields..."},
)

# End the generation with output
tracker.end(
    observation_id=observation_id,
    output={"fields": {"amount": "$500", "date": "2024-01-15"}},
    usage={
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "total_tokens": 200,
    },
)

# Update trace with final results
tracker.update_trace(
    trace_id=trace_id,
    output={"status": "success", "fields_extracted": 5},
    metadata={"latency_seconds": 2.5},
)
```

### Error handling

```python
try:
    # LLM call that might fail
    response = llm_client.complete(prompt)
    tracker.end(observation_id, output=response)
except Exception as e:
    # Track the error
    tracker.error(
        observation_id=observation_id,
        error=e,
        level="ERROR",
    )
```

### Spans for non-LLM operations

```python
# Track non-LLM operations (database queries, API calls, etc.)
span_id = tracker.span(
    trace_id=trace_id,
    name="fetch-document",
    input={"document_id": "doc-123"},
)

# ... perform operation ...

tracker.end(
    observation_id=span_id,
    output={"document_size": 1024},
)
```

## Database schema

### PostgreSQL (metadata only)

The `llm_raw_events` table stores essential tracking metadata:

```sql
CREATE TABLE llm_raw_events (
    -- Core identifiers
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    s3_key VARCHAR(500) NOT NULL,

    -- Model & Provider (for analytics)
    model_name VARCHAR(100),
    model_provider VARCHAR(50),

    -- Token metrics (for usage/cost analytics)
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,

    -- Performance metrics (for latency analytics)
    duration_ms INTEGER,
    time_to_first_token_ms INTEGER,

    -- Cost tracking
    cost_usd DECIMAL(10, 6),

    -- Context
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    tags JSONB,

    -- Status & timestamps
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,

    CONSTRAINT valid_status CHECK (status IN ('pending', 'processed', 'failed'))
);

-- Indexes for common queries
CREATE INDEX idx_llm_events_status ON llm_raw_events(status);
CREATE INDEX idx_llm_events_trace_id ON llm_raw_events(trace_id);
CREATE INDEX idx_llm_events_created_at ON llm_raw_events(created_at);
CREATE INDEX idx_llm_events_model ON llm_raw_events(model_name, model_provider);
CREATE INDEX idx_llm_events_user ON llm_raw_events(user_id);
CREATE INDEX idx_llm_events_session ON llm_raw_events(session_id);
```

Run the migration:

```bash
psql -d marie -f config/psql/schema/048_llm_tracking.sql
```

## Running the worker

The worker can run as part of the main service or as a standalone process.

### Embedded worker

When `worker.enabled: true` in config, the worker runs in a background thread:

```python
from marie.utils.server_runtime import setup_llm_tracking

# Called during server startup
setup_llm_tracking(config["llm_tracking"], config.get("storage"))
```

### Standalone worker

Run the worker as a separate process for better isolation and scaling:

```bash
python -m marie.llm_tracking.worker
```

Or with custom config:

```python
from marie.llm_tracking.worker import LLMTrackingWorker
from marie.llm_tracking.config import configure_from_yaml

# Load config
configure_from_yaml(llm_tracking_config, storage_config)

# Start worker
worker = LLMTrackingWorker()
worker.run()  # Blocking
```

## Retry and error handling

The worker implements automatic retry with limits:

- **Max retries**: 3 attempts per event
- **Retry tracking**: Uses `x-retry-count` message header
- **Failed events**: Marked as `status='failed'` in PostgreSQL after max retries
- **Poison messages**: Removed from queue after max retries (not requeued forever)

## Requeuing failed events

Use the requeue script to recover failed events:

```bash
# Dry run - see what would be requeued
python -m marie.llm_tracking.scripts.requeue_failed \
    --config config/service/marie-gateway.yml \
    --status failed \
    --limit 100 \
    --dry-run

# Actually requeue events
python -m marie.llm_tracking.scripts.requeue_failed \
    --config config/service/marie-gateway.yml \
    --status failed \
    --limit 100
```

Options:

| Option | Description | Default |
|--------|-------------|---------|
| `--config, -c` | Path to YAML config file | Required |
| `--status` | Event status to query (`failed` or `pending`) | `failed` |
| `--limit` | Maximum events to requeue | `100` |
| `--dry-run` | Print what would be done without making changes | `false` |
| `--verbose, -v` | Enable verbose logging | `false` |

## Analytics queries

With metadata in PostgreSQL, you can run analytics queries without accessing S3:

```sql
-- Token usage by model
SELECT
    model_name,
    SUM(total_tokens) as total_tokens,
    SUM(cost_usd) as total_cost
FROM llm_raw_events
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY model_name
ORDER BY total_tokens DESC;

-- Latency percentiles by model
SELECT
    model_name,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms) as p50,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99
FROM llm_raw_events
WHERE status = 'processed'
GROUP BY model_name;

-- Error rates by model
SELECT
    model_name,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'failed') / COUNT(*), 2) as error_rate
FROM llm_raw_events
GROUP BY model_name;

-- Usage by user
SELECT
    user_id,
    COUNT(*) as requests,
    SUM(total_tokens) as total_tokens,
    SUM(cost_usd) as total_cost
FROM llm_raw_events
WHERE user_id IS NOT NULL
GROUP BY user_id
ORDER BY total_cost DESC
LIMIT 20;
```

## Cleanup

Old processed events can be cleaned up automatically:

```python
from marie.llm_tracking.storage.postgres import PostgresStorage

postgres = PostgresStorage()
postgres.start()

# Delete processed events older than 30 days
deleted = postgres.cleanup_old_events(days=30)
print(f"Deleted {deleted} old events")
```

For S3 payloads, use S3 lifecycle policies:

```json
{
  "Rules": [
    {
      "ID": "Delete old LLM tracking payloads",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "llm-events/"
      },
      "Expiration": {
        "Days": 90
      }
    }
  ]
}
```

## Troubleshooting

### Events stuck in pending status

Check if the worker is running and connected to RabbitMQ:

```bash
# Check RabbitMQ queue
rabbitmqctl list_queues name messages consumers

# Check worker logs
journalctl -u marie-llm-worker -f
```

### Events marked as failed

Query failed events to see error messages:

```sql
SELECT id, trace_id, event_type, error_message, created_at
FROM llm_raw_events
WHERE status = 'failed'
ORDER BY created_at DESC
LIMIT 20;
```

Common causes:
- S3 payload not found (check s3_key)
- ClickHouse connection issues
- Invalid event data

### Worker not processing events

1. Verify RabbitMQ connection and vhost configuration
2. Check that PostgreSQL and S3 are accessible
3. Ensure the exchange and queue exist
4. Check for poison messages blocking the queue

### High latency in tracking

If tracking adds noticeable latency:
- Ensure RabbitMQ is running locally or has low network latency
- Consider using the async exporter for non-blocking operation
- Check S3 upload performance

## Exporter types

### Console exporter (development)

For local development and debugging:

```yaml
llm_tracking:
  enabled: true
  exporter: console
```

Events are logged to console instead of being persisted.

### RabbitMQ exporter (production)

For production use with full durability:

```yaml
llm_tracking:
  enabled: true
  exporter: rabbitmq
  # ... rabbitmq, postgres, s3 config required
```

:::warning
The RabbitMQ exporter requires both PostgreSQL and S3 to be configured. The tracker will fail to start if storage is not properly configured.
:::

## Integration with batch processing

The LLM tracking system integrates with Marie-AI's batch processor:

```python
from marie.engine import BatchProcessor

processor = BatchProcessor(
    model="gpt-4",
    tracking_enabled=True,  # Enable tracking
    project_id="batch-processing",
)

# Traces are automatically created for batch jobs
results = await processor.process(documents)
```

Each batch job creates:
- A trace for the overall batch
- Observations for each LLM generation
- Automatic error tracking for failures
- Final trace update with success/failure counts
