---
sidebar_position: 3
---

# Flows

A Flow orchestrates Executors into a processing pipeline to accomplish document processing tasks. Documents "flow" through the pipeline and are processed by each Executor in sequence.

## What is a flow

A Flow is a set of connected Executors that work together to process documents. You can think of a Flow as:

- An interface to configure and launch your microservice architecture
- A way to chain document processing steps (OCR, classification, extraction)
- A service that exposes your pipeline through gRPC, HTTP, or WebSocket APIs

Each Flow also launches a Gateway service that exposes all Executors through a unified API.

## Why use a flow

- **Connect microservices**: Flows connect Executors to build document processing pipelines with proper client/server interfaces
- **Scale independently**: Scale each Executor independently based on processing requirements
- **Cloud-native deployment**: Export Flows to Docker Compose or Kubernetes for production deployment
- **Protocol flexibility**: Serve via gRPC, HTTP, or WebSocket protocols

## Creating flows

The simplest Flow is an empty one that contains only a Gateway:

```python
from marie import Flow

f = Flow()
```

Or define it in YAML:

```yaml
jtype: Flow
```

:::tip
For production, define Flows in YAML files. This separates configuration from code and makes deployments easier to manage.
:::

## Minimum working example

```python
from marie import Flow, Executor, requests
from docarray import DocList, BaseDoc

class ProcessDocument(Executor):
    @requests(on='/process')
    def process(self, docs: DocList[BaseDoc], **kwargs) -> DocList[BaseDoc]:
        for doc in docs:
            print(f'Processing document: {doc}')
        return docs

f = Flow().add(name='processor', uses=ProcessDocument)

with f:
    f.post(on='/process', inputs=BaseDoc(), return_type=DocList[BaseDoc])
```

### Flow as a service

**Server:**

```python
from marie import Flow, Executor, requests
from docarray import DocList, BaseDoc

class ProcessDocument(Executor):
    @requests(on='/process')
    def process(self, docs: DocList[BaseDoc], **kwargs) -> DocList[BaseDoc]:
        return docs

f = Flow(port=12345).add(name='processor', uses=ProcessDocument)

with f:
    f.block()  # Keep the Flow running
```

**Client:**

```python
from marie import Client
from docarray import DocList, BaseDoc

c = Client(port=12345)
c.post(on='/process', inputs=BaseDoc(), return_type=DocList[BaseDoc])
```

### Load from YAML

`flow.yml`:

```yaml
jtype: Flow
version: '1'
with:
  protocol: [grpc, http]
  port: [54321, 54322]
executors:
  - name: processor
    uses: ProcessDocument
    py_modules: executor.py
```

`executor.py`:

```python
from marie import Executor, requests
from docarray import DocList
from docarray.documents import TextDoc

class ProcessDocument(Executor):
    @requests
    def process(self, docs: DocList[TextDoc], **kwargs) -> DocList[TextDoc]:
        for doc in docs:
            doc.text = 'processed'
        return docs
```

Load and run:

```python
from marie import Flow
from docarray import DocList
from docarray.documents import TextDoc

f = Flow.load_config('flow.yml')

with f:
    result = f.post(on='/', inputs=TextDoc(text='hello'))
```

:::warning
The `with f:` statement starts the Flow. Exiting the block stops the Flow and all its Executors. Use `try...except` blocks if you need to handle exceptions without stopping the Flow.
:::

## Start and stop

### Using context manager

```python
from marie import Flow

f = Flow()

with f:
    # Flow is running
    f.block()
# Flow is stopped
```

### Using CLI

```bash
marie flow --uses flow.yml
```

### Manual start/stop

```python
from marie import Flow

f = Flow()
f.start()
# ... use the Flow ...
f.close()
```

### Serve forever

```python
from marie import Flow

f = Flow()

with f:
    f.block()  # Blocks until interrupted
```

### Serve until an event

```python
from marie import Flow
import threading

def start_flow(stop_event):
    f = Flow()
    with f:
        f.block(stop_event=stop_event)

e = threading.Event()
t = threading.Thread(target=start_flow, args=(e,))
t.start()

# Later: stop the Flow
e.set()
```

## Adding executors to flows

Add Executors to a Flow using the `add()` method:

```python
from marie import Flow

f = Flow().add(name='executor1').add(name='executor2')
```

Equivalent YAML:

```yaml
jtype: Flow
executors:
  - name: executor1
  - name: executor2
```

### Define executor with `uses`

The `uses` parameter specifies the Executor type:

```python
from marie import Flow

# Use a Python class
f = Flow().add(uses=MyExecutor)

# Use a YAML config
f = Flow().add(uses='executor-config.yml')

# Use a Docker container
f = Flow().add(uses='docker://marieai/my-executor')
```

### Configure executors

Override Executor configuration when adding to a Flow:

```python
from marie import Flow

f = Flow().add(
    uses='MyExecutor',
    py_modules=['executor.py'],
    uses_with={'model_path': '/models/v1', 'device': 'cuda'},
    uses_metas={'name': 'document-processor'},
    workspace='/data/workspace'
)
```

| Parameter | Description |
| --------- | ----------- |
| `uses` | Executor class, YAML path, or Docker image |
| `py_modules` | Python files to import |
| `uses_with` | Arguments passed to `__init__()` |
| `uses_metas` | Metadata (name, description) |
| `workspace` | Working directory for the Executor |

## Document processing pipelines

Marie-AI Flows are optimized for document processing workflows. A typical pipeline includes:

1. **Document ingestion**: Load documents from storage
2. **Pre-processing**: Burst pages, clean images
3. **OCR**: Extract text from images
4. **Classification**: Determine document type
5. **Extraction**: Extract structured data
6. **Post-processing**: Validate and store results

Example pipeline:

```python
from marie import Flow

f = (
    Flow(protocol='grpc', port=54321)
    .add(name='loader', uses='DocumentLoader')
    .add(name='ocr', uses='OCRExecutor')
    .add(name='classifier', uses='ClassificationExecutor')
    .add(name='extractor', uses='ExtractionExecutor')
)
```

YAML equivalent:

```yaml
jtype: Flow
version: '1'
with:
  protocol: grpc
  port: 54321
executors:
  - name: loader
    uses: DocumentLoader
  - name: ocr
    uses: OCRExecutor
  - name: classifier
    uses: ClassificationExecutor
  - name: extractor
    uses: ExtractionExecutor
```

## Define custom topologies

Flows support complex, non-sequential topologies using the `needs` parameter:

```python
from marie import Flow

f = (
    Flow()
    .add(name='preprocess', uses=PreprocessExecutor)
    .add(name='ocr', uses=OCRExecutor, needs='preprocess')
    .add(name='classify', uses=ClassifyExecutor, needs='preprocess')
    .add(name='merge', uses=MergeExecutor, needs=['ocr', 'classify'])
)
```

This creates a parallel pipeline where `ocr` and `classify` run simultaneously after `preprocess`, and `merge` combines their results.

### Visualize the flow

```python
f.plot('flow.svg')
```

Or from the terminal:

```bash
marie export flowchart flow.yml flow.svg
```

## Gateway configuration

The Gateway handles incoming requests and routes them to Executors.

:::tip
For comprehensive Gateway documentation including MarieGateway features, REST API endpoints, and advanced configuration, see the [Gateway guide](./gateway.md).
:::

### Multi-protocol support

```python
from marie import Flow

f = Flow(
    protocol=['grpc', 'http', 'websocket'],
    port=[54321, 54322, 54323]
)
```

YAML:

```yaml
jtype: Flow
with:
  protocol: [grpc, http, websocket]
  port: [54321, 54322, 54323]
```

### CORS configuration

For HTTP endpoints that need cross-origin access:

```python
from marie import Flow

f = Flow(protocol='http', port=54321, cors=True)
```

## Scaling flows

### Replicas

Scale an Executor by adding replicas:

```python
from marie import Flow

f = Flow().add(name='processor', uses=MyExecutor, replicas=3)
```

YAML:

```yaml
jtype: Flow
executors:
  - name: processor
    uses: MyExecutor
    replicas: 3
```

### Shards

Partition data across multiple instances:

```python
from marie import Flow

f = Flow().add(name='indexer', uses=MyIndexer, shards=2)
```

## YAML configuration

### Flow YAML structure

```yaml
jtype: Flow
version: '1'
with:
  # Flow/Gateway arguments
  protocol: http
  port: 54321
  cors: true
executors:
  - name: executor1
    uses: MyExecutor
    py_modules:
      - executor.py
    uses_with:
      param1: value1
    replicas: 2
  - name: executor2
    uses: config.yml
    needs: executor1
```

### Fields reference

| Field | Description |
| ----- | ----------- |
| `jtype` | Always "Flow" |
| `version` | Configuration version |
| `with` | Flow and Gateway arguments |
| `executors` | List of Executor configurations |

### Executor fields

| Field | Description |
| ----- | ----------- |
| `name` | Unique identifier for the Executor |
| `uses` | Executor class, YAML, or Docker image |
| `py_modules` | Python files to import |
| `uses_with` | Constructor arguments |
| `uses_metas` | Metadata |
| `needs` | Dependencies (other Executor names) |
| `replicas` | Number of replicas |
| `shards` | Number of shards |

## Export for deployment

### Docker Compose

```python
from marie import Flow

f = Flow().add(uses='MyExecutor')
f.to_docker_compose_yaml('docker-compose.yml')
```

Or from CLI:

```bash
marie export docker-compose flow.yml docker-compose.yml
```

### Kubernetes

```python
from marie import Flow

f = Flow().add(uses='MyExecutor')
f.to_kubernetes_yaml('k8s-config/')
```

Or from CLI:

```bash
marie export kubernetes flow.yml ./k8s-config
```

The generated files can be deployed with:

```bash
kubectl apply -R -f k8s-config/
```

## Monitoring and health checks

### Check if flow is ready

```python
from marie import Flow

f = Flow()

with f:
    if f.is_flow_ready():
        print('Flow is ready to process requests')
```

### Health check endpoints

When using HTTP protocol, the Gateway exposes health endpoints:

- `GET /status` - Returns Flow status
- `GET /ready` - Readiness probe for Kubernetes

## Flow methods reference

| Method | Description |
| ------ | ----------- |
| `add()` | Add an Executor to the Flow |
| `start()` | Start the Flow |
| `close()` | Stop and close the Flow |
| `block()` | Block execution until interrupted |
| `post()` | Send requests to the Flow |
| `plot()` | Visualize the Flow |
| `to_docker_compose_yaml()` | Export to Docker Compose |
| `to_kubernetes_yaml()` | Export to Kubernetes |
| `is_flow_ready()` | Check if Flow is ready |

## Best practices

1. **Use YAML for production**: Keep configuration separate from code
2. **Name your Executors**: Use meaningful names for easier debugging
3. **Start small**: Begin with a simple pipeline and add complexity as needed
4. **Monitor resource usage**: Use replicas for CPU-bound and shards for memory-bound operations
5. **Test locally first**: Validate your Flow before deploying to production

## Next steps

- Learn about [Executors](./executor.md) for building processing components
- See [Architecture overview](./architecture-overview.md) for the big picture
- Explore [Deployment guides](../getting-started/deployment/index.md) for production setup
