---
sidebar_position: 4
---

# Client

The Client enables you to send documents to a running Flow for processing. It supports three networking protocols: gRPC, HTTP, and WebSocket, with optional TLS encryption.

## Connecting to a flow

To connect to a Flow, create a Client with matching protocol, host, and port:

```python
from marie import Client

# Connect to a local Flow
client = Client(host='localhost', port=54321, protocol='grpc')

# Or use a URI scheme
client = Client(host='grpc://localhost:54321')
```

### Connection parameters

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `host` | Server hostname or IP | `localhost` |
| `port` | Server port | `80` (or `443` with TLS) |
| `protocol` | Communication protocol (`grpc`, `http`, `websocket`) | `grpc` |
| `tls` | Enable TLS encryption | `False` |

### Using URI schemes

You can specify connection details using URI schemes:

```python
from marie import Client

# Without TLS
Client(host='http://my-server:54321')
Client(host='grpc://my-server:54321')
Client(host='ws://my-server:54321')

# With TLS
Client(host='https://my-server:54321')
Client(host='grpcs://my-server:54321')
Client(host='wss://my-server:54321')
```

Or use keyword arguments:

```python
from marie import Client

# Without TLS
Client(host='my-server', port=54321, protocol='http')

# With TLS
Client(host='my-server', port=54321, protocol='http', tls=True)
```

## Sending requests

Use the `post()` method to send documents to a Flow endpoint:

```python
from marie import Client
from docarray import DocList, BaseDoc

class TextDoc(BaseDoc):
    text: str = ''

client = Client(host='localhost', port=54321)

# Send a single document
doc = TextDoc(text='Hello, Marie-AI!')
result = client.post(on='/process', inputs=doc, return_type=DocList[TextDoc])

# Send multiple documents
docs = DocList[TextDoc]([
    TextDoc(text='Document 1'),
    TextDoc(text='Document 2'),
])
results = client.post(on='/process', inputs=docs, return_type=DocList[TextDoc])
```

### Input formats

The Client accepts various input formats:

```python
from marie import Client
from docarray import DocList, BaseDoc

class MyDoc(BaseDoc):
    content: str = ''

client = Client(port=54321)

# Single document
client.post('/endpoint', inputs=MyDoc(content='hello'))

# List of documents
client.post('/endpoint', inputs=[MyDoc(content='hello'), MyDoc(content='world')])

# DocList
docs = DocList[MyDoc]([MyDoc(content='hello'), MyDoc(content='world')])
client.post('/endpoint', inputs=docs)

# Generator (memory efficient for large datasets)
def doc_generator():
    for i in range(1000):
        yield MyDoc(content=f'document {i}')

client.post('/endpoint', inputs=doc_generator())

# No input (for status endpoints)
client.post('/status')
```

### Return types

Specify the expected return type using `return_type`:

```python
from marie import Client
from docarray import DocList, BaseDoc

class InputDoc(BaseDoc):
    text: str = ''

class OutputDoc(BaseDoc):
    text: str = ''
    word_count: int = 0

client = Client(port=54321)

# Specify return type
results = client.post(
    on='/process',
    inputs=InputDoc(text='hello world'),
    return_type=DocList[OutputDoc]
)

for doc in results:
    print(f'Text: {doc.text}, Words: {doc.word_count}')
```

### Sending parameters

Pass additional parameters to the Executor:

```python
from marie import Client
from docarray import DocList, BaseDoc

class MyDoc(BaseDoc):
    content: str = ''

client = Client(port=54321)

results = client.post(
    on='/extract',
    inputs=MyDoc(content='invoice data'),
    parameters={
        'job_id': 'job-123',
        'layout': 'invoice',
        'confidence_threshold': 0.8,
    },
    return_type=DocList[MyDoc]
)
```

## Batching requests

When processing large datasets, documents are internally batched into requests. Control batch size with `request_size`:

```python
from marie import Client
from docarray import DocList, BaseDoc

class MyDoc(BaseDoc):
    content: str = ''

client = Client(port=54321)

# Create many documents
docs = DocList[MyDoc]([MyDoc(content=f'doc {i}') for i in range(10000)])

# Process in batches of 100 (default)
results = client.post('/process', inputs=docs, return_type=DocList[MyDoc])

# Process in smaller batches (better for memory)
results = client.post('/process', inputs=docs, request_size=50, return_type=DocList[MyDoc])
```

## Async client

For asynchronous applications, use the async Client:

```python
import asyncio
from marie import Client
from docarray import DocList, BaseDoc

class MyDoc(BaseDoc):
    content: str = ''

async def process_documents():
    client = Client(port=54321, asyncio=True)

    # Async generator for inputs
    async def doc_generator():
        for i in range(100):
            yield MyDoc(content=f'document {i}')
            await asyncio.sleep(0.01)

    # Process asynchronously
    async for response in client.post('/process', inputs=doc_generator(), request_size=10):
        print(f'Received batch with {len(response)} documents')

# Run async client
asyncio.run(process_documents())
```

### Async client in executors

Use async clients when calling external services from within an Executor:

```python
from marie import Executor, requests, Client
from docarray import DocList, BaseDoc

class MyDoc(BaseDoc):
    content: str = ''

class ForwardingExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Client(host='grpc://other-service:54321', asyncio=True)

    @requests(on='/forward')
    async def forward(self, docs: DocList[MyDoc], **kwargs) -> DocList[MyDoc]:
        # Forward to another service
        results = await self.client.post('/process', inputs=docs, return_type=DocList[MyDoc])
        return results
```

## Callback functions

Use callbacks to process responses as they arrive:

```python
from marie import Client
from docarray import DocList, BaseDoc

class MyDoc(BaseDoc):
    content: str = ''

def on_done(response):
    """Called when a batch completes successfully."""
    print(f'Processed {len(response.docs)} documents')

def on_error(response):
    """Called when an error occurs."""
    print(f'Error: {response}')

def on_always(response):
    """Called after every batch, regardless of success or failure."""
    print('Batch complete')

client = Client(port=54321)

# With callbacks, post() returns None
client.post(
    on='/process',
    inputs=DocList[MyDoc]([MyDoc(content=f'doc {i}') for i in range(100)]),
    request_size=10,
    on_done=on_done,
    on_error=on_error,
    on_always=on_always,
)
```

:::tip
Callbacks are more memory-efficient for large datasets because responses are processed immediately and freed from memory, rather than accumulated.
:::

## Target specific executors

Send requests to specific Executors in a Flow using `target_executor`:

```python
from marie import Client
from docarray import DocList, BaseDoc

class MyDoc(BaseDoc):
    content: str = ''

client = Client(port=54321)

# Target executor by exact name
results = client.post(
    on='/process',
    inputs=MyDoc(content='hello'),
    target_executor='ocr-executor',
    return_type=DocList[MyDoc]
)

# Target executors matching a pattern (regex)
results = client.post(
    on='/process',
    inputs=MyDoc(content='hello'),
    target_executor='extract-*',  # Matches extract-v1, extract-v2, etc.
    return_type=DocList[MyDoc]
)
```

## Health checks

Check if the Flow is ready to receive requests:

```python
from marie import Client

client = Client(port=54321)

# Check readiness
if client.is_flow_ready():
    print('Flow is ready')
    results = client.post('/process', inputs=docs)
else:
    print('Flow is not ready')
```

## Profiling latency

Measure network latency before sending data:

```python
from marie import Client

client = Client(host='grpc://my-server:54321')
client.profiling()
```

Output:

```text
 Roundtrip  24ms  100%
├──  Client-server network  17ms  71%
└──  Server  7ms  29%
    ├──  Gateway-executors network  0ms  0%
    ├──  executor0  5ms  71%
    └──  executor1  2ms  29%
```

## Response ordering

By default, responses may arrive out of order due to parallel processing. To preserve order:

```python
from marie import Client
from docarray import DocList, BaseDoc

class MyDoc(BaseDoc):
    content: str = ''

client = Client(port=54321)

docs = DocList[MyDoc]([MyDoc(content=f'doc-{i}') for i in range(100)])

# Force ordered responses
results = client.post(
    on='/process',
    inputs=docs,
    results_in_order=True,
    return_type=DocList[MyDoc]
)

# Results are now in the same order as inputs
for input_doc, output_doc in zip(docs, results):
    assert input_doc.content == output_doc.content
```

## gRPC compression

Enable compression for gRPC connections:

```python
from marie import Client

client = Client(protocol='grpc', port=54321)

# Enable gzip compression
results = client.post('/process', inputs=docs, compression='gzip')

# Or deflate compression
results = client.post('/process', inputs=docs, compression='deflate')
```

## gRPC channel options

Customize gRPC channel settings:

```python
from marie import Client

client = Client(
    protocol='grpc',
    port=54321,
    grpc_channel_options={
        'grpc.max_send_message_length': 100 * 1024 * 1024,  # 100MB
        'grpc.max_receive_message_length': 100 * 1024 * 1024,
        'grpc.keepalive_time_ms': 10000,
    }
)
```

Default gRPC options:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `grpc.max_send_message_length` | -1 (unlimited) | Max outgoing message size |
| `grpc.max_receive_message_length` | -1 (unlimited) | Max incoming message size |
| `grpc.keepalive_time_ms` | 9999 | Keepalive ping interval |
| `grpc.keepalive_timeout_ms` | 4999 | Keepalive timeout |

## HTTP requests with curl

When your Flow uses HTTP protocol, you can send requests with curl:

```bash
# Send a document for processing
curl -X POST http://localhost:54321/process \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"content": "Document to process"}
    ]
  }'

# With parameters
curl -X POST http://localhost:54321/extract \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"content": "Invoice #12345"}
    ],
    "parameters": {
      "job_id": "job-123",
      "layout": "invoice"
    }
  }'

# Check status
curl http://localhost:54321/status
```

## Error handling

Handle errors gracefully:

```python
from marie import Client
from marie.excepts import BadClientInput, ConnectionError
from docarray import DocList, BaseDoc

class MyDoc(BaseDoc):
    content: str = ''

client = Client(port=54321)

try:
    results = client.post('/process', inputs=MyDoc(content='test'), return_type=DocList[MyDoc])
except ConnectionError as e:
    print(f'Could not connect to Flow: {e}')
except BadClientInput as e:
    print(f'Invalid input: {e}')
except Exception as e:
    print(f'Unexpected error: {e}')
```

## Complete example

Here's a complete example showing common Client patterns:

```python
from marie import Client
from docarray import DocList, BaseDoc
from typing import Optional

# Define document types
class DocumentInput(BaseDoc):
    content: str = ''
    doc_type: str = 'unknown'

class DocumentOutput(BaseDoc):
    content: str = ''
    doc_type: str = ''
    extracted_data: Optional[dict] = None
    confidence: float = 0.0

def main():
    # Create client
    client = Client(host='localhost', port=54321, protocol='http')

    # Check if service is ready
    if not client.is_flow_ready():
        print('Service not ready, exiting')
        return

    # Prepare documents
    documents = DocList[DocumentInput]([
        DocumentInput(content='Invoice #123 Total: $500', doc_type='invoice'),
        DocumentInput(content='Dear Customer, Thank you...', doc_type='letter'),
    ])

    # Process with parameters
    results = client.post(
        on='/extract',
        inputs=documents,
        parameters={
            'job_id': 'batch-001',
            'confidence_threshold': 0.7,
        },
        return_type=DocList[DocumentOutput]
    )

    # Handle results
    for doc in results:
        print(f'Type: {doc.doc_type}')
        print(f'Confidence: {doc.confidence:.2%}')
        print(f'Extracted: {doc.extracted_data}')
        print('---')

if __name__ == '__main__':
    main()
```

## Next steps

- Build custom [Executors](./executor.md) for document processing
- Orchestrate with [Flows](./flow.md)
- See the [Quickstart](../getting-started/quickstart.md) for a complete tutorial
