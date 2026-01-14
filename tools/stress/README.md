# Marie-AI Stress Testing Tools

This directory contains stress testing tools for testing the Marie gateway and networking components.

## Toolsgit 

### 1. `gateway_stresser.py`

A stress tester for the Marie gateway that supports both gRPC and HTTP protocols.

#### Features

- **Multi-protocol support**: gRPC and HTTP (via aiohttp)
- **Configurable load**: Concurrency, request rate, duration
- **Authentication**: Bearer token support (default API key included)
- **Metrics collection**: Latency percentiles (p50, p95, p99), success rates, throughput
- **Health checks**: Automatic connectivity testing before stress test

#### Usage

```bash
# Basic HTTP test (uses /api/v1/invoke and default API key)
python gateway_stresser.py --protocol http --http-port 51000

# Basic gRPC test
python gateway_stresser.py --protocol grpc --gateway-port 52000

# High load test
python gateway_stresser.py --protocol http --http-port 51000 \
    --concurrency 50 --request-rate 100 --duration 120

# Test specific endpoint
python gateway_stresser.py --protocol http --http-port 51000 --endpoint /extract

# With custom request parameters
python gateway_stresser.py --protocol http --http-port 51000 \
    --parameters '{"invoke_action": {"action_type": "command", "command": "job", "action": "submit", "name": "test"}}'

# Compare gRPC vs HTTP performance
python gateway_stresser.py --protocol grpc --gateway-port 52000 --duration 60
python gateway_stresser.py --protocol http --http-port 51000 --duration 60
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gateway-host` | `localhost` | Gateway host |
| `--gateway-port` | `52000` | Gateway gRPC port |
| `--http-port` | same as gateway-port | Gateway HTTP port |
| `--protocol` | `grpc` | Protocol: `grpc`, `http`, `websocket` |
| `--endpoint` | `/api/v1/invoke` | Endpoint to test |
| `--concurrency` | `10` | Number of concurrent workers |
| `--request-rate` | `10.0` | Target requests per second |
| `--timeout` | `30.0` | Request timeout in seconds |
| `--duration` | `60.0` | Test duration in seconds |
| `--warmup` | `5.0` | Warmup period in seconds |
| `--batch-size` | `1` | Documents per request |
| `--api-key` | (default key) | API key for authentication |
| `--parameters` | `None` | JSON string of request parameters |
| `--target-executor` | `None` | Target executor name |
| `-v, --verbose` | `False` | Enable verbose logging |

#### Output

The tool provides real-time progress updates and a final report:

```
======================================================================
GATEWAY STRESS TEST REPORT
======================================================================

Test Duration: 60.0 seconds

--- Request Summary ---
Total Requests: 600
Successful: 598
Failed: 2
Timeouts: 0
Success Rate: 99.67%
Throughput: 10.00 req/s

--- Latency Statistics (ms) ---
Min: 5.23
Max: 125.67
Avg: 15.42
P50: 12.34
P95: 45.67
P99: 89.12
Std Dev: 12.45

======================================================================
RESULT: EXCELLENT - Gateway performing well under load
======================================================================
```

---

### 2. `networking_stresser.py`

A comprehensive networking stress tester with chaos engineering capabilities.

#### Features

- **Executor simulation**: Simulates N executor servers
- **Chaos controller**: Randomly cycles executors up/down
- **Multiple test modes**: Circuit breaker, load balancer, chaos, flood testing
- **Real-time metrics**: Periodic reporting during test execution

#### Usage

```bash
# Full integration test with 5 executors
python networking_stresser.py --duration 120

# Test with 10 executors and aggressive chaos
python networking_stresser.py --num-executors 10 \
    --chaos-interval-min 2 --chaos-interval-max 5

# Test circuit breaker behavior
python networking_stresser.py --mode circuit_breaker_test

# High request rate testing
python networking_stresser.py --request-rate 100 --mode request_flood

# Disable chaos (stable environment)
python networking_stresser.py --no-chaos --duration 60
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gateway-host` | `localhost` | Gateway host |
| `--gateway-port` | `52000` | Gateway port |
| `--num-executors` | `5` | Number of simulated executors |
| `--executor-base-port` | `54000` | Base port for executors |
| `--no-chaos` | `False` | Disable chaos (executor up/down) |
| `--chaos-interval-min` | `5` | Min seconds between chaos events |
| `--chaos-interval-max` | `15` | Max seconds between chaos events |
| `--chaos-down-duration-min` | `3` | Min seconds executor stays down |
| `--chaos-down-duration-max` | `10` | Max seconds executor stays down |
| `--chaos-max-down-ratio` | `0.5` | Max ratio of executors that can be down |
| `--request-rate` | `10` | Requests per second |
| `--duration` | `60` | Test duration in seconds |
| `--mode` | `full_integration` | Test mode |

#### Test Modes

- `circuit_breaker_test`: Tests circuit breaker behavior under failure conditions
- `load_balancer_test`: Tests load distribution across replicas
- `chaos_test`: Aggressive chaos testing with frequent executor failures
- `full_integration`: Complete integration test with all features
- `request_flood`: High request rate stress testing

---

## Prerequisites

1. **Marie gateway must be running**:
   ```bash
   marie server --start --uses config/service/marie.yml
   ```

2. **Python dependencies**:
   ```bash
   pip install aiohttp grpcio
   ```

## Typical Test Workflow

1. **Start the gateway**:
   ```bash
   marie server --start --uses /mnt/data/marie-ai/config/service/extract/marie-extract-4.0.0.yml
   ```

2. **Run basic connectivity test**:
   ```bash
   python tools/stress/gateway_stresser.py --protocol http --http-port 51000 --duration 10
   ```

3. **Run load test**:
   ```bash
   python tools/stress/gateway_stresser.py --protocol http --http-port 51000 \
       --concurrency 50 --request-rate 100 --duration 300
   ```

4. **Compare protocols**:
   ```bash
   # gRPC test
   python tools/stress/gateway_stresser.py --protocol grpc --gateway-port 52000 --duration 60

   # HTTP test
   python tools/stress/gateway_stresser.py --protocol http --http-port 51000 --duration 60
   ```

## Interpreting Results

| Success Rate | Verdict | Meaning |
|--------------|---------|---------|
| >= 99% | EXCELLENT | Gateway performing well under load |
| >= 95% | GOOD | Minor issues detected |
| >= 90% | FAIR | Some reliability concerns |
| < 90% | POOR | Significant issues detected |

### Key Metrics to Watch

- **P95/P99 Latency**: High values indicate tail latency issues
- **Throughput**: Should match or exceed target request rate
- **Error Types**: Connection errors vs timeout errors vs HTTP errors
- **Success Rate**: Should be > 95% for production readiness
