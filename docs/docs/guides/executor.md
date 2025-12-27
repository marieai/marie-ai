---
sidebar_position: 2
---

# Executors

An Executor is the fundamental building block in Marie-AI. It is a self-contained microservice that processes documents and can be exposed via gRPC, HTTP, or WebSocket protocols.

## What is an executor

An Executor is a Python class that:

1. Subclasses from `marie.Executor` (or `MarieExecutor` for document processing)
2. Contains methods decorated with `@requests` that handle specific endpoints
3. Processes documents using DocArray's `DocList` and document types
4. Can be orchestrated in a Flow for complex document processing pipelines

Executors follow a microservice architecture pattern, allowing you to scale, deploy, and manage each processing step independently.

## Creating your first executor

To create an Executor, define a Python class that inherits from `Executor` and decorate methods with `@requests`:

```python
from marie import Executor, requests
from docarray import DocList, BaseDoc

class MyExecutor(Executor):
    @requests
    def process(self, docs: DocList[BaseDoc], **kwargs) -> DocList[BaseDoc]:
        for doc in docs:
            # Process each document
            pass
        return docs
```

### Project structure

A typical Executor project has the following structure:

```text
MyExecutor/
├── executor.py      # Main executor logic
├── config.yml       # Executor configuration
├── requirements.txt # Python dependencies
└── README.md        # Documentation
```

### Constructor

If your Executor needs initialization logic, implement `__init__` with `**kwargs` and call `super().__init__(**kwargs)`:

```python
from marie import Executor

class MyExecutor(Executor):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.device = device
        # Initialize your model or resources here
```

:::note
The `kwargs` parameter is required because Marie-AI injects runtime configuration like `metas`, `requests` mappings, and `runtime_args` when the Executor runs inside a Flow.
:::

### Destructor

Override the `close` method to clean up resources when the Executor shuts down:

```python
from marie import Executor

class MyExecutor(Executor):
    def close(self):
        # Clean up resources
        if hasattr(self, 'model'):
            del self.model
        print('Executor closed')
```

Marie-AI ensures `close` is called when the Executor terminates, whether in a Flow or standalone deployment.

## Adding endpoints

Methods decorated with `@requests` are exposed as network endpoints.

### Basic endpoint binding

```python
from marie import Executor, requests

class MyExecutor(Executor):
    @requests(on='/process')
    def process_documents(self, docs, **kwargs):
        # Handle /process endpoint
        return docs

    @requests(on='/status')
    def get_status(self, **kwargs):
        return {'status': 'healthy'}
```

### Default endpoint

A method decorated with `@requests` without `on=` becomes the default handler for unmatched endpoints:

```python
from marie import Executor, requests

class MyExecutor(Executor):
    @requests
    def default_handler(self, docs, **kwargs):
        # Handles any endpoint not explicitly defined
        return docs
```

### Multiple endpoints

You can bind a single method to multiple endpoints:

```python
from marie import Executor, requests

class MyExecutor(Executor):
    @requests(on=['/extract', '/process'])
    def handle_extraction(self, docs, **kwargs):
        # Handles both /extract and /process
        return docs
```

### Async endpoints

Both synchronous and asynchronous methods are supported:

```python
from marie import Executor, requests
import asyncio

class MyExecutor(Executor):
    @requests(on='/async-process')
    async def async_process(self, docs, **kwargs):
        await asyncio.sleep(0.1)  # Simulate async operation
        return docs
```

## Document type binding

Marie-AI uses DocArray for document handling. You can specify input and output types using type annotations:

```python
from marie import Executor, requests
from marie.api.docs import AssetKeyDoc
from docarray import DocList, BaseDoc
from docarray.typing import AnyTensor
from typing import Optional

class InputDoc(BaseDoc):
    text: str = ''

class OutputDoc(BaseDoc):
    text: str = ''
    embedding: Optional[AnyTensor] = None

class MyExecutor(Executor):
    @requests(on='/embed')
    def embed(self, docs: DocList[InputDoc], **kwargs) -> DocList[OutputDoc]:
        results = DocList[OutputDoc]()
        for doc in docs:
            results.append(OutputDoc(
                text=doc.text,
                embedding=self.model.encode(doc.text)
            ))
        return results
```

### Marie-AI document types

Marie-AI provides specialized document types for document processing:

| Type | Description |
| ---- | ----------- |
| `AssetKeyDoc` | Document reference with asset key and page numbers |
| `MarieDoc` | Image document with tags for OCR processing |
| `BatchableMarieDoc` | OCR results with words and bounding boxes |
| `StorageDoc` | Document for persistence layer |
| `OutputDoc` | Job status and response document |

Example using `AssetKeyDoc`:

```python
from marie import Executor, requests
from marie.api.docs import AssetKeyDoc
from docarray import DocList

class DocumentProcessor(Executor):
    @requests(on='/extract')
    async def extract(
        self,
        docs: DocList[AssetKeyDoc],
        parameters: dict,
        **kwargs
    ):
        doc = docs[0]
        asset_key = doc.asset_key  # S3 or local path
        pages = doc.pages          # Page numbers to process
        # Process the document
        return {'status': 'success'}
```

### Parameters

Executor methods can receive additional parameters beyond documents:

```python
from marie import Executor, requests
from pydantic import BaseModel, Field

class ExtractionParams(BaseModel):
    job_id: str = Field(..., description='Unique job identifier')
    layout: str = Field(default='default', description='Layout configuration')
    confidence_threshold: float = Field(default=0.5)

class MyExecutor(Executor):
    @requests(on='/extract')
    def extract(
        self,
        docs,
        parameters: ExtractionParams,
        **kwargs
    ):
        job_id = parameters.job_id
        layout = parameters.layout
        # Use parameters in processing
        return docs
```

## Document processing executors

Marie-AI provides specialized base classes for document processing tasks.

### MarieExecutor

`MarieExecutor` extends the base `Executor` with:

- GPU health monitoring via NVML
- PyTorch CUDA probing
- Toast event notifications
- Storage integration
- Fault markers for error recovery

```python
from marie.executor.marie_executor import MarieExecutor
from marie import requests
from docarray import DocList
from docarray.documents import TextDoc

class MyDocumentExecutor(MarieExecutor):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.device = device

    @requests(on='/process')
    async def process(
        self,
        docs: DocList[TextDoc],
        parameters: dict,
        **kwargs
    ) -> DocList[TextDoc]:
        # Process documents with GPU support
        return docs
```

### DocumentAnnotatorExecutor

For document extraction and annotation tasks:

```python
from marie.executor.extract.document_annotator_executor import DocumentAnnotatorExecutor
from marie.api.docs import AssetKeyDoc
from marie import requests
from docarray import DocList

class MyAnnotator(DocumentAnnotatorExecutor):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model = self.load_model(model_path)

    @requests(on='/annotate')
    async def annotate(
        self,
        docs: DocList[AssetKeyDoc],
        parameters: dict,
        **kwargs
    ):
        # Document annotation logic
        return {'status': 'success'}
```

## Storage integration

Marie-AI executors can persist data using the `StorageMixin`:

```python
from marie.executor.marie_executor import MarieExecutor
from marie.executor.mixin import StorageMixin

class PersistentExecutor(MarieExecutor, StorageMixin):
    def __init__(self, storage: dict = None, **kwargs):
        kwargs['storage'] = storage
        super().__init__(**kwargs)

        # Setup storage from config
        if storage and 'psql' in storage:
            sconf = storage['psql']
            self.setup_storage(
                storage_enabled=sconf.get('enabled', False),
                storage_conf=sconf,
                asset_tracking_enabled=sconf.get('asset_tracking_enabled', True)
            )
```

Storage configuration in YAML:

```yaml
jtype: PersistentExecutor
with:
  storage:
    psql:
      enabled: true
      hostname: localhost
      port: 5432
      database: marie
      username: marie
      password: ${POSTGRES_PASSWORD}
      asset_tracking_enabled: true
```

## Configuration (YAML)

Executors can be configured via YAML files:

```yaml
jtype: MyExecutor
with:
  model_path: /models/extraction
  device: cuda
  num_workers: 4
py_modules:
  - executor.py
metas:
  name: document-processor
  description: Processes documents for extraction
```

### Configuration keywords

| Keyword | Description |
| ------- | ----------- |
| `jtype` | Python class name of the Executor |
| `with` | Arguments passed to `__init__()` |
| `py_modules` | Python files to import |
| `metas` | Metadata (name, description, etc.) |

### Using configuration

Load an Executor from YAML:

```python
from marie import Deployment

# Using YAML config
dep = Deployment(uses='config.yml')

# Or specify inline
dep = Deployment(
    uses='MyExecutor',
    uses_with={'model_path': '/models/v1', 'device': 'cuda'}
)
```

## Executor attributes

When running inside a Flow, Executors have access to runtime attributes:

### workspace

Path reserved for this Executor instance:

```python
class MyExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f'Workspace: {self.workspace}')
        # Store files in self.workspace
```

### metas

Metadata about the Executor:

```python
class MyExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f'Name: {self.metas.name}')
        print(f'Description: {self.metas.description}')
```

### runtime_args

Runtime information (replicas, shards, etc.):

```python
class MyExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f'Replica ID: {self.runtime_args.shard_id}')
        print(f'Total replicas: {self.runtime_args.replicas}')
```

## Exception handling

Exceptions in `@requests` methods propagate to the client:

```python
from marie import Executor, requests

class MyExecutor(Executor):
    @requests(on='/process')
    def process(self, docs, **kwargs):
        if not docs:
            raise ValueError('No documents provided')
        return docs
```

For GPU-related errors, `MarieExecutor` provides automatic fault detection and process termination for recovery.

## Health checks

Marie-AI executors support health check endpoints:

```python
from marie.executor.marie_executor import MarieExecutor
from marie import requests
from docarray import DocList
from docarray.documents import TextDoc

class HealthyExecutor(MarieExecutor):
    @requests(on='/status')
    async def status(
        self,
        docs: DocList[TextDoc],
        **kwargs
    ) -> DocList[TextDoc]:
        # Returns health status as JSON
        return DocList[TextDoc]([
            TextDoc(text='{"healthy": true, "gpu": "available"}')
        ])
```

## Containerization

Package your Executor as a Docker container:

```dockerfile
FROM marieai/marie:latest

COPY executor.py config.yml requirements.txt ./
RUN pip install -r requirements.txt

ENTRYPOINT ["marie", "executor", "--uses", "config.yml"]
```

Build and run:

```bash
docker build -t my-executor .
docker run --gpus all -p 54321:54321 my-executor
```

## Best practices

1. **Single responsibility**: Each Executor should handle one specific task
2. **Stateless when possible**: Store state externally for easier scaling
3. **Use async for I/O**: Async methods improve throughput for I/O-bound operations
4. **Handle errors gracefully**: Return meaningful error responses
5. **Document your endpoints**: Use type annotations for clear API contracts
6. **Test locally first**: Use Executors as regular Python classes for unit testing

```python
# Local testing without Flow
executor = MyExecutor(model_path='/models/v1')
result = executor.process(docs=test_docs, parameters={})
```

## Next steps

- Learn about [Flows](./flow.md) to orchestrate multiple Executors
- See [Architecture overview](./architecture-overview.md) for the big picture
- Explore [Deployment guides](../getting-started/deployment/index.md) for production setup
