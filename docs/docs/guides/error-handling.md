---
sidebar_position: 8
---

# Error handling

Handle errors gracefully in Marie-AI deployments. This guide covers exception handling, retry policies, timeout configuration, and debugging techniques.

## Overview

Errors in Marie-AI can occur at different levels:

| Level | Examples | Handling |
|-------|----------|----------|
| Executor | Processing failures, model errors | Retry or fail job |
| Network | Connection timeouts, unreachable services | Automatic retry |
| Gateway | Request validation, routing errors | Return error response |
| Scheduler | Job expiration, dependency failures | State transition |

```text
┌─────────────────────────────────────────────────────────────┐
│                      Error Flow                             │
│                                                             │
│   Client Request                                            │
│        │                                                    │
│        ▼                                                    │
│   ┌─────────┐    timeout/error    ┌─────────────────────┐  │
│   │ Gateway │ ──────────────────▶ │ Retry or Error      │  │
│   └────┬────┘                     │ Response            │  │
│        │                          └─────────────────────┘  │
│        │ route                                              │
│        ▼                                                    │
│   ┌──────────┐   exception        ┌─────────────────────┐  │
│   │ Executor │ ─────────────────▶ │ Retry, Fail, or     │  │
│   └──────────┘                    │ Terminate           │  │
│                                   └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Executor errors

### Initialization errors

If an Executor's `__init__` method raises an exception, the Flow cannot start:

```python
from marie import Executor

class MyExecutor(Executor):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = load_model(model_path)
```

When initialization fails:
- The Executor runtime raises the exception
- The Flow throws a `RuntimeFailToStart` exception
- The client receives an error

**Best practice:** Validate configuration early and provide clear error messages.

### Request handler errors

Errors in `@requests` methods are captured and returned to the client:

```python
from marie import Executor, requests
from docarray import DocList, BaseDoc

class ProcessingExecutor(Executor):
    @requests(on='/process')
    def process(self, docs: DocList[BaseDoc], **kwargs) -> DocList[BaseDoc]:
        for doc in docs:
            try:
                result = self.process_document(doc)
            except ProcessingError as e:
                # Log the error and continue with other documents
                self.logger.error(f"Failed to process document: {e}")
                doc.tags['error'] = str(e)
        return docs
```

Error behavior by protocol:

| Protocol | Behavior |
|----------|----------|
| gRPC | Error in response, stream continues |
| HTTP | Error in response body |
| WebSocket | Error message, connection stays open |

### Terminate on fatal errors

Some errors put an Executor in an unusable state. Configure automatic termination:

```yaml
# flow.yml
executors:
  - name: processor
    uses: MyExecutor
    exit_on_exceptions:
      - RuntimeError
      - OutOfMemoryError
      - CUDAError
```

Or in Python:

```python
from marie import Flow

f = Flow().add(
    uses=MyExecutor,
    exit_on_exceptions=['RuntimeError', 'OutOfMemoryError']
)
```

When a matching exception occurs:
1. The Executor terminates gracefully
2. In Kubernetes, the pod restarts automatically
3. The autoscaler creates a replacement pod

## Network errors

### Retry policy

When the Gateway can't reach an Executor, it retries according to a policy:

```yaml
# flow.yml
jtype: Flow
with:
  retries: 3  # Number of retry attempts
```

Or in Python:

```python
from marie import Flow

f = Flow(retries=3).add(uses=MyExecutor)
```

**Default retry behavior:**

| Deployment | Policy |
|------------|--------|
| Local | Try each replica once, minimum 3 attempts total |
| Kubernetes (no mesh) | 3 attempts to same replica |
| Kubernetes (with mesh) | 3 attempts distributed across replicas |

### Timeout configuration

Set timeouts to prevent requests from hanging:

```yaml
# flow.yml
jtype: Flow
with:
  timeout_send: 60000  # 60 seconds in milliseconds
```

Or in Python:

```python
from marie import Flow

f = Flow(timeout_send=60000).add(uses=MyExecutor)
```

**Guidelines:**
- Set higher timeouts for GPU-intensive operations
- Consider the slowest expected processing time
- Add buffer for network latency

```python
# For slow ML operations
f = Flow(timeout_send=300000)  # 5 minutes
```

### Connection errors

When retries are exhausted, errors are reported to the client:

| Error Type | gRPC Code | HTTP Code | Description |
|------------|-----------|-----------|-------------|
| Connection failed | 14 (UNAVAILABLE) | 503 | Executor unreachable |
| Timeout | 4 (DEADLINE_EXCEEDED) | 504 | Request timed out |
| Internal error | 13 (INTERNAL) | 500 | Unexpected failure |

## Rate limiting

Prevent overload with prefetch limits:

```yaml
# flow.yml
jtype: Flow
with:
  prefetch: 10  # Max concurrent requests per client
```

Or in Python:

```python
from marie import Flow

f = Flow().config_gateway(prefetch=10).add(uses=MyExecutor)
```

**Guidelines:**

| Scenario | Prefetch Setting |
|----------|------------------|
| Fast executors | 100-1000 (default) |
| Slow executors | 1-10 |
| Memory-intensive | 1-5 |
| Unknown | Start with 1, increase gradually |

:::warning
For slow executors processing large data, set `prefetch=1` to prevent out-of-memory errors.
:::

## Job-level error handling

### Retry configuration

Configure retries at the job level:

```python
from marie.scheduler.models import WorkInfo

job = WorkInfo(
    name="extract",
    retry_limit=3,        # Maximum retry attempts
    retry_delay=5,        # Initial delay in seconds
    retry_backoff=True,   # Enable exponential backoff
    # ...
)
```

With exponential backoff:
- Retry 1: 5 seconds
- Retry 2: 10 seconds
- Retry 3: 20 seconds

### Job failure states

Jobs transition through states on failure:

```text
ACTIVE ──▶ RETRY ──▶ ACTIVE ──▶ ... ──▶ FAILED
           (delay)
```

| Transition | Condition |
|------------|-----------|
| ACTIVE → RETRY | Failure with retries remaining |
| RETRY → ACTIVE | Retry delay elapsed |
| ACTIVE → FAILED | No retries remaining |

### Handling dependency failures

In DAGs, dependency failures can cascade:

```python
# Job with dependencies
job = WorkInfo(
    name="extract",
    dag_id="my-dag",
    dependencies=["job-ocr-001"],  # Must complete first
    # ...
)
```

When a dependency fails:
- Dependent jobs may be skipped
- DAG state reflects the failure
- Notifications are sent

## Error responses

### REST API errors

```json
{
  "status": "error",
  "message": "Processing failed: Document format not supported",
  "code": "PROCESSING_ERROR",
  "details": {
    "job_id": "abc-123",
    "executor": "extract"
  }
}
```

### gRPC errors

```python
from marie import Client

try:
    client = Client(port=54322)
    response = client.post('/process', inputs=docs)
except ConnectionError as e:
    print(f"Connection failed: {e}")
except TimeoutError as e:
    print(f"Request timed out: {e}")
```

## Debugging

### Enable debug logging

```yaml
# Environment variable
MARIE_LOG_LEVEL: DEBUG
```

Or in configuration:

```yaml
logging:
  level: DEBUG
```

### Debug breakpoints

Standard Python breakpoints don't work inside Executors running in a Flow context. Use `epdb` instead:

```python
from marie import Executor, requests

class DebugExecutor(Executor):
    @requests
    def process(self, docs, **kwargs):
        import epdb; epdb.set_trace()  # Works in Flow context
        return docs
```

Install with: `pip install epdb`

:::note
Regular `breakpoint()` or `pdb.set_trace()` won't work when Executors run in separate processes.
:::

### Inspect job failures

Query failed jobs:

```bash
curl http://localhost:54322/api/jobs/FAILED
```

Check job details:

```sql
SELECT id, name, state, data->'error' as error
FROM marie_scheduler.job
WHERE state = 'failed'
ORDER BY completed_on DESC
LIMIT 10;
```

### Common debugging commands

```bash
# Check executor logs
kubectl logs -l app=marie-executor --tail=100

# Check gateway logs
kubectl logs -l app=marie-gateway --tail=100

# Describe failing pod
kubectl describe pod <pod-name>

# Check recent events
kubectl get events --sort-by='.lastTimestamp' | tail -20
```

## Best practices

### 1. Fail fast on unrecoverable errors

```python
class MyExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.validate_config():
            raise ValueError("Invalid configuration")
```

### 2. Use appropriate retry limits

```python
# Transient errors (network): higher retry limit
network_job = WorkInfo(retry_limit=5, retry_backoff=True, ...)

# Deterministic errors (invalid input): lower retry limit
validation_job = WorkInfo(retry_limit=1, ...)
```

### 3. Log errors with context

```python
class MyExecutor(Executor):
    @requests
    def process(self, docs, **kwargs):
        for doc in docs:
            try:
                self.process_doc(doc)
            except Exception as e:
                self.logger.error(
                    f"Processing failed",
                    extra={
                        'doc_id': doc.id,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
                )
                raise
```

### 4. Set realistic timeouts

```yaml
# Consider: model loading + processing + buffer
timeout_send: 120000  # 2 minutes for GPU operations
```

### 5. Monitor error rates

```promql
# Error rate query
rate(marie_jobs_failed_total[5m]) / rate(marie_jobs_total[5m]) * 100
```

Set alerts for high error rates:

```yaml
- alert: HighErrorRate
  expr: rate(marie_jobs_failed_total[5m]) / rate(marie_jobs_total[5m]) > 0.1
  for: 5m
  labels:
    severity: warning
```

## Next steps

- [Job lifecycle](../getting-started/job-management/job-lifecycle.md) - Job state transitions
- [Troubleshooting](../getting-started/deployment/troubleshooting.md) - Common issues
- [Observability](../getting-started/deployment/observability.md) - Monitoring and alerting
