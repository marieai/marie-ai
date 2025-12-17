---
sidebar_position: 5
---

# Gateway

The Gateway is the entry point for all client requests to a Marie-AI Flow. It receives requests over the network, routes them to the appropriate Executors, and returns the final response to the client.

## What is the gateway

Every Flow has a Gateway component that:

- Receives incoming requests from clients
- Routes requests to Executors in the processing pipeline
- Handles responses and returns them to clients
- Supports multiple protocols (gRPC, HTTP, WebSocket)
- Provides health check endpoints for monitoring

In most cases, the Gateway is automatically configured when you initialize a Flow. However, you can explicitly configure it for advanced use cases.

## Gateway protocols

Marie-AI supports three communication protocols:

| Protocol | Best for | Features |
|----------|----------|----------|
| **gRPC** | High-performance, production | Binary protocol, streaming, low latency |
| **HTTP** | REST APIs, web integration | JSON payloads, browser-friendly, OpenAPI |
| **WebSocket** | Real-time, bidirectional | Persistent connections, streaming |

### Configure protocol in Python

```python
from marie import Flow, Executor, requests
from docarray import DocList, BaseDoc

class MyExecutor(Executor):
    @requests
    def process(self, docs: DocList[BaseDoc], **kwargs) -> DocList[BaseDoc]:
        return docs

# gRPC (default)
f = Flow(protocol='grpc', port=54321).add(uses=MyExecutor)

# HTTP
f = Flow(protocol='http', port=54321).add(uses=MyExecutor)

# WebSocket
f = Flow(protocol='websocket', port=54321).add(uses=MyExecutor)
```

### Configure protocol in YAML

```yaml
jtype: Flow
with:
  protocol: http
  port: 54321
executors:
  - name: processor
    uses: MyExecutor
```

### Enable multiple protocols

You can serve a Flow over multiple protocols simultaneously:

```python
from marie import Flow

f = Flow(
    protocol=['grpc', 'http', 'websocket'],
    port=[54321, 54322, 54323]
)
```

YAML equivalent:

```yaml
jtype: Flow
with:
  protocol:
    - grpc
    - http
    - websocket
  port:
    - 54321
    - 54322
    - 54323
```

:::warning
When using multiple protocols, specify one port per protocol. The ports are matched to protocols in order.
:::

## Health checks

The Gateway exposes health check endpoints for monitoring and orchestration systems like Kubernetes.

### gRPC health check

When using gRPC, the Gateway implements the [standard gRPC health check protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md).

Check health using grpcurl:

```bash
grpcurl -plaintext localhost:54321 grpc.health.v1.Health/Check
```

Response:

```json
{
  "status": "SERVING"
}
```

### HTTP health check

For HTTP protocol, query the root endpoint:

```bash
curl http://localhost:54321/
```

A successful response (empty JSON `{}`) indicates the Gateway is healthy.

You can also use the `/status` endpoint for detailed information:

```bash
curl http://localhost:54321/status
```

Response includes version information and environment details:

```json
{
  "marie": {
    "version": "3.0.0",
    "docarray": "0.40.0",
    "python": "3.10.0",
    "platform": "Linux"
  }
}
```

### WebSocket health check

For WebSocket, test by establishing a connection to the root endpoint. A successful connection indicates the Gateway is healthy.

## TLS encryption

Enable TLS encryption between the Gateway and clients:

```python
from marie import Flow

f = Flow(
    port=54321,
    ssl_certfile='path/to/certfile.crt',
    ssl_keyfile='path/to/keyfile.key'
)
```

:::note
TLS encrypts traffic between clients and the Gateway. Internal communication between Executors uses gRPC and is not encrypted by default.
:::

## CORS configuration

For HTTP endpoints accessed from web browsers, enable CORS:

```python
from marie import Flow

f = Flow(protocol='http', port=54321, cors=True)
```

YAML:

```yaml
jtype: Flow
with:
  protocol: http
  port: 54321
  cors: true
```

## Rate limiting

Control request flow with the `prefetch` parameter:

```python
from marie import Flow

f = Flow(prefetch=100)  # Limit to 100 in-flight requests per client
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `prefetch` | Max in-flight requests per client | 1000 |
| `timeout_send` | Request timeout in milliseconds | None |

Setting `prefetch=0` disables rate limiting.

## gRPC options

Customize gRPC server behavior:

```python
from marie import Flow

f = Flow(
    grpc_server_options={
        'grpc.max_send_message_length': 100 * 1024 * 1024,  # 100MB
        'grpc.max_receive_message_length': 100 * 1024 * 1024,
        'grpc.keepalive_time_ms': 10000,
    }
)
```

Default gRPC options:

| Option | Default | Description |
|--------|---------|-------------|
| `grpc.max_send_message_length` | -1 (unlimited) | Max message size to send |
| `grpc.max_receive_message_length` | -1 (unlimited) | Max message size to receive |
| `grpc.keepalive_time_ms` | 9999 | Keepalive ping interval |
| `grpc.keepalive_timeout_ms` | 4999 | Keepalive timeout |

## In-Flow compression

Enable compression for internal communication between Executors:

```python
from marie import Flow

f = Flow(compression='gzip')  # Options: 'gzip', 'deflate', or None
```

---

## MarieGateway

MarieGateway extends the base Gateway with production features for document processing:

- **Service discovery**: ETCD-based executor registration and discovery
- **Job scheduling**: PostgreSQL-backed job queue with priorities and SLAs
- **Real-time events**: Server-Sent Events (SSE) for job status updates
- **Capacity management**: Slot-based resource allocation
- **Query planners**: Dynamic query routing and orchestration

### Configuration

MarieGateway requires additional configuration for its extended features:

```yaml
jtype: Flow
with:
  protocol: [grpc, http]
  port: [54321, 54322]
gateway:
  uses: MarieGateway
  with:
    # PostgreSQL for job scheduler
    kv_store_kwargs:
      provider: postgresql
      hostname: localhost
      port: 5432
      username: marie
      password: ${POSTGRES_PASSWORD}
      database: marie

    job_scheduler_kwargs:
      provider: postgresql
      hostname: localhost
      port: 5432
      username: marie
      password: ${POSTGRES_PASSWORD}
      database: marie

    # ETCD for service discovery
    discovery_host: localhost
    discovery_port: 2379
    discovery_service_name: marie
```

### REST API reference

MarieGateway exposes additional REST endpoints for management and monitoring.

#### Job management

**Submit a job**

```bash
curl "http://localhost:54322/job/submit?text=test"
```

Response:

```json
{
  "result": "job-uuid-12345"
}
```

**List jobs**

```bash
# List all jobs
curl http://localhost:54322/api/jobs

# Filter by state
curl http://localhost:54322/api/jobs/SCHEDULED
```

Response:

```json
{
  "status": "OK",
  "result": [
    {
      "id": "job-uuid-12345",
      "name": "extract",
      "state": "SCHEDULED",
      "priority": 0,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

**Stop a job**

```bash
curl http://localhost:54322/api/jobs/{job_id}/stop
```

**Delete a job**

```bash
curl -X DELETE http://localhost:54322/api/jobs/{job_id}
```

#### Deployment information

**Get all deployments**

```bash
curl http://localhost:54322/api/deployments
```

Response:

```json
{
  "status": "OK",
  "result": {
    "deployments": {...},
    "deployment_nodes": {...}
  }
}
```

**Get deployment nodes**

```bash
curl http://localhost:54322/api/deployment-nodes
```

**Get deployment status**

```bash
curl http://localhost:54322/api/deployment-status
```

Response includes desired state and actual status for each deployment.

#### Capacity management

**Get capacity information**

```bash
curl http://localhost:54322/api/capacity
```

Response:

```json
{
  "status": "OK",
  "result": {
    "slots": [
      {
        "name": "ocr.gpu",
        "capacity": 4,
        "target": 4,
        "used": 2,
        "available": 2,
        "holders": ["worker-1", "worker-2"],
        "notes": ""
      }
    ],
    "totals": {...}
  }
}
```

#### Query planners

**List registered planners**

```bash
curl http://localhost:54322/api/planners
```

Response:

```json
{
  "planners": [
    {
      "id": "extract-v1",
      "name": "Document Extraction",
      "version": "1.0.0",
      "description": "Standard extraction pipeline"
    }
  ],
  "total": 1
}
```

**Get planner by ID**

```bash
curl http://localhost:54322/api/planners/{planner_id}
```

**Register a new planner**

```bash
curl -X POST http://localhost:54322/api/planners \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom-extractor",
    "plan": {...},
    "description": "Custom extraction workflow",
    "version": "1.0.0"
  }'
```

**Unregister a planner**

```bash
curl -X DELETE http://localhost:54322/api/planners/{planner_id}
```

#### Command invocation

**Invoke a command**

Requires authentication:

```bash
curl -X POST http://localhost:54322/api/v1/invoke \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "header": {},
    "parameters": {
      "command": "extract",
      "asset_key": "s3://bucket/document.pdf"
    }
  }'
```

Response:

```json
{
  "header": {},
  "parameters": {
    "job_id": "job-uuid-12345",
    "status": "submitted"
  },
  "data": null
}
```

#### Real-time events (SSE)

Subscribe to real-time events using Server-Sent Events:

**Subscribe to all events**

```bash
curl -H "Authorization: Bearer your-token" \
  http://localhost:54322/sse/all
```

**Subscribe to tenant-specific events**

```bash
curl -H "Authorization: Bearer your-token" \
  http://localhost:54322/sse/{api_key}
```

Events are streamed in SSE format:

```text
event: job.status
data: {"job_id": "12345", "status": "completed", "progress": 100}

event: capacity.update
data: {"slot": "ocr.gpu", "available": 3}
```

#### Health and debug

**Health check**

```bash
curl "http://localhost:54322/check?text=ping"
```

**Debug information**

```bash
curl http://localhost:54322/api/debug
```

**Reset active DAGs**

```bash
curl -X POST http://localhost:54322/api/debug/reset-dags
```

### Authentication

MarieGateway supports token-based authentication for protected endpoints:

```python
from marie.auth.auth_bearer import TokenBearer

# Endpoints decorated with TokenBearer require authentication
@app.api_route(
    path="/api/v1/invoke",
    methods=["POST"],
    dependencies=[Depends(TokenBearer())]
)
```

Include the bearer token in requests:

```bash
curl -H "Authorization: Bearer your-api-token" \
  http://localhost:54322/api/v1/invoke
```

### Service discovery

MarieGateway uses ETCD for service discovery. Executors register themselves when they start, and the Gateway watches for changes:

```yaml
gateway:
  with:
    discovery_host: localhost
    discovery_port: 2379
    discovery_service_name: marie
```

The Gateway automatically:

- Discovers new Executors when they come online
- Removes Executors when they go offline
- Updates routing tables for request distribution

## Deployment considerations

### Kubernetes

When deploying to Kubernetes, configure health checks:

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: gateway
      livenessProbe:
        httpGet:
          path: /
          port: 54322
        initialDelaySeconds: 10
        periodSeconds: 5
      readinessProbe:
        httpGet:
          path: /status
          port: 54322
        initialDelaySeconds: 5
        periodSeconds: 3
```

### Load balancing

For high availability, deploy multiple Gateway replicas behind a load balancer. Each Gateway instance connects to the same ETCD cluster for consistent service discovery.

## Next steps

- Learn about [Flows](./flow.md) for orchestrating pipelines
- See [Client](./client.md) for connecting to the Gateway
- Explore [Architecture overview](./architecture-overview.md) for the big picture
- Review [Deployment guides](../getting-started/deployment/index.md) for production setup
