# Mock Executor Configuration

This directory contains configuration files for testing Marie-AI with mock executors that simulate various execution behaviors.

## Configuration Files

### marie-mock-4.0.0.yml
Basic single mock executor configuration. Use this for:
- Simple integration testing
- Basic functionality validation
- Development and debugging

**Default settings:**
- `process_time: 3.0` - 3 second processing time
- `failure_rate: 0.0` - No failures
- `failure_mode: exception` - Exception-based failures

### marie-mock-scheduler-test.yml
Advanced configuration with 8 different mock executors for comprehensive scheduler testing. Use this for:
- Testing parallel subgraph execution
- SLA enforcement testing
- Error handling and retry logic
- Resource allocation testing
- Job distribution performance testing

**Executors:**
| Name | Process Time | Failure Rate | Purpose |
|------|--------------|--------------|---------|
| mock_executor_a | 1.0s | 0% | Fast, reliable baseline |
| mock_executor_b | 2.0s | 0% | Medium speed, reliable |
| mock_executor_c | 3.0s | 0% | Standard processing |
| mock_executor_d | 5.0s | 0% | Slow, SLA boundary testing |
| mock_executor_e | 1.5s | 10% | Light flakiness |
| mock_executor_f | 3.0s | 20% | Moderate flakiness |
| mock_executor_g | 8.0s | 0% | Very slow, timeout testing |
| mock_executor_h | 2.0s | 50% | Heavy error testing |

## Mock Executor Parameters

### Constructor Parameters

All mock executors support these parameters in the `with` section:

```yaml
with:
  process_time: 3.0        # Base processing time in seconds
  failure_rate: 0.0        # Probability of failure (0.0 to 1.0)
  failure_mode: exception  # Type of failure simulation
```

**Failure Modes:**
- `exception` - Raises RuntimeError immediately
- `timeout` - Simulates timeout by sleeping 10x longer then raises TimeoutError
- `random` - Randomly chooses between RuntimeError, ValueError, or ConnectionError

### Per-Request Parameter Overrides

You can override executor-level settings for individual requests by passing parameters in the request:

```python
parameters = {
    "process_time": 5.0,  # Override processing time
    "failure_rate": 0.3,  # Override failure rate
    "failure_mode": "timeout",  # Override failure mode
    "force_fail": True,  # Force a failure
    "randomize_time": True,  # Add ±50% randomness to process_time
}
```

## Usage Examples

### Starting Marie with Mock Configuration

```bash
# Basic mock executor
marie server --start --uses config/service/mock/marie-mock-4.0.0.yml

# Scheduler testing configuration
marie server --start --uses config/service/mock/marie-mock-scheduler-test.yml
```

### Testing with Mock Query Plans

The mock executors work seamlessly with the mock query plans defined in `tests/integration/scheduler/mock_query_plans.py`:

```python
from marie.query_planner.base import PlannerInfo, QueryPlanRegistry
from marie.job.job_manager import generate_job_id

# Create planner info
planner_info = PlannerInfo(name="mock_parallel_subgraphs", base_id=generate_job_id())

# Get the query plan
planner_func = QueryPlanRegistry.get("mock_parallel_subgraphs")
plan = planner_func(planner_info)

# Execute the plan (scheduler will distribute tasks to mock executors)
```

### Testing Scenarios

#### 1. SLA Testing
Use `mock_executor_a` (fast) vs `mock_executor_g` (slow) to test SLA enforcement:
```yaml
# Job with tight SLA (should complete with mock_executor_a, fail with mock_executor_g)
sla_config:
  soft_sla_ms: 2000
  hard_sla_ms: 4000
```

#### 2. Error Recovery Testing
Use `mock_executor_h` (50% failure rate) to test retry logic and error handling:
```yaml
# Configure retry behavior
retry_config:
  max_retries: 3
  backoff_multiplier: 2
```

#### 3. Parallel Execution Testing
Use `marie-mock-scheduler-test.yml` with `mock_parallel_subgraphs` query plan to test:
- Concurrent task execution across multiple executors
- Resource contention
- Load balancing
- Job distribution algorithms

#### 4. Realistic Simulation
Override parameters per-request for realistic variability:
```python
# Simulate variable execution times
parameters = {
    "randomize_time": True,  # ±50% variance
    "failure_rate": 0.05,  # 5% failure rate
}
```

## Testing with Different Query Plans

### mock_simple
**Nodes:** 3 (linear)
**Best config:** `marie-mock-4.0.0.yml`
**Purpose:** Basic functionality testing

### mock_medium
**Nodes:** 7 (with parallel branch)
**Best config:** `marie-mock-4.0.0.yml`
**Purpose:** Parallel execution testing

### mock_complex
**Nodes:** 12 (parallel + sequential)
**Best config:** `marie-mock-scheduler-test.yml`
**Purpose:** Complex DAG execution

### mock_with_subgraphs
**Nodes:** ~13 (nested subgraphs)
**Best config:** `marie-mock-scheduler-test.yml`
**Purpose:** Subgraph pattern testing

### mock_parallel_subgraphs
**Nodes:** ~23 (3 parallel subgraphs)
**Best config:** `marie-mock-scheduler-test.yml`
**Purpose:** Comprehensive scheduler stress testing

## Environment Variables

Ensure these environment variables are set:

```bash
# PostgreSQL (required for scheduler)
export DB_HOSTNAME=localhost

# RabbitMQ (required for event tracking)
export RABBIT_MQ_HOSTNAME=localhost
export RABBIT_MQ_PORT=5672
export RABBIT_MQ_USERNAME=guest
export RABBIT_MQ_PASSWORD=guest

# S3 (optional for storage)
export S3_ENDPOINT_URL=http://localhost:9000
export S3_ACCESS_KEY_ID=minioadmin
export S3_SECRET_ACCESS_KEY=minioadmin
export S3_BUCKET_NAME=marie-test
export S3_REGION=us-east-1
```

## Troubleshooting

### Mock executor not processing
- Check logs for async/await issues
- Verify executor is registered and started
- Check endpoint configuration matches `/document/process`

### Scheduler not distributing jobs
- Verify PostgreSQL connection
- Check `scheduler.enabled: True` in config
- Ensure executors are registered in service discovery

### Failures not occurring as expected
- Verify `failure_rate` is between 0.0 and 1.0
- Check `failure_mode` is valid: exception, timeout, or random
- Use `force_fail: True` in parameters to guarantee failure

## Advanced Configuration

### Custom Executor Configuration

Create custom mock executor configurations for specific testing needs:

```yaml
executors:
  - name: custom_mock
    uses:
      jtype: IntegrationExecutorMock
      with:
        # Very fast for high-throughput testing
        process_time: 0.1
        failure_rate: 0.0

        # Or very flaky for stress testing
        process_time: 2.0
        failure_rate: 0.8
        failure_mode: random
```

### Per-Environment Configuration

Use environment variable substitution for dynamic configuration:

```yaml
with:
  process_time: ${{ ENV.MOCK_PROCESS_TIME }}
  failure_rate: ${{ ENV.MOCK_FAILURE_RATE }}
  failure_mode: ${{ ENV.MOCK_FAILURE_MODE }}
```

Then set environment variables:
```bash
export MOCK_PROCESS_TIME=3.0
export MOCK_FAILURE_RATE=0.1
export MOCK_FAILURE_MODE=exception
```
