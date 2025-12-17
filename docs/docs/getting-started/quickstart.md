---
sidebar_position: 2
---

# Quickstart

This guide walks you through building your first Marie-AI document processing pipeline in under 10 minutes.

## Prerequisites

Before you begin, ensure you have:

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-capable GPU for accelerated processing

## Step 1: Install Marie-AI

Create a virtual environment and install Marie-AI:

```bash
# Create and activate virtual environment
python -m venv marie-env
source marie-env/bin/activate  # On Windows: marie-env\Scripts\activate

# Install Marie-AI
pip install marie-ai
```

Verify the installation:

```bash
marie --version
```

## Step 2: Create your first executor

An Executor is a microservice that processes documents. Create a file called `my_executor.py`:

```python
from marie import Executor, requests
from docarray import DocList, BaseDoc

class TextDoc(BaseDoc):
    text: str = ''
    processed: bool = False

class SimpleProcessor(Executor):
    """A simple executor that processes text documents."""

    @requests(on='/process')
    def process(self, docs: DocList[TextDoc], **kwargs) -> DocList[TextDoc]:
        for doc in docs:
            # Transform the text
            doc.text = doc.text.upper()
            doc.processed = True
            print(f'Processed: {doc.text}')
        return docs

    @requests(on='/status')
    def status(self, **kwargs):
        return {'status': 'healthy', 'executor': 'SimpleProcessor'}
```

## Step 3: Create a flow

A Flow orchestrates Executors into a pipeline. Create a file called `my_flow.py`:

```python
from marie import Flow
from my_executor import SimpleProcessor

# Create a Flow with your Executor
f = Flow(port=54321, protocol='http').add(
    name='processor',
    uses=SimpleProcessor
)

if __name__ == '__main__':
    with f:
        print('Flow is ready! Send requests to http://localhost:54321')
        f.block()  # Keep running until interrupted
```

## Step 4: Start the service

Run your Flow:

```bash
python my_flow.py
```

You should see output indicating the Flow is ready:

```text
────────────────────────── Flow is ready to serve! ──────────────────────────
╭────────────── 🔗 Endpoint ───────────────╮
│  ⛓     Protocol                   HTTP  │
│  🏠       Local           0.0.0.0:54321  │
╰──────────────────────────────────────────╯
```

## Step 5: Send requests

Open a new terminal and create a client script called `client.py`:

```python
from marie import Client
from docarray import DocList, BaseDoc

class TextDoc(BaseDoc):
    text: str = ''
    processed: bool = False

# Create a client
c = Client(host='localhost', port=54321, protocol='http')

# Create input documents
docs = DocList[TextDoc]([
    TextDoc(text='hello world'),
    TextDoc(text='marie-ai is awesome'),
])

# Send to the /process endpoint
results = c.post(on='/process', inputs=docs, return_type=DocList[TextDoc])

# Print results
for doc in results:
    print(f'Text: {doc.text}, Processed: {doc.processed}')
```

Run the client:

```bash
python client.py
```

Output:

```text
Text: HELLO WORLD, Processed: True
Text: MARIE-AI IS AWESOME, Processed: True
```

## Step 6: Use YAML configuration

For production, define your Flow in YAML. Create `flow.yml`:

```yaml
jtype: Flow
version: '1'
with:
  port: 54321
  protocol: http
executors:
  - name: processor
    uses: SimpleProcessor
    py_modules:
      - my_executor.py
```

Run with the CLI:

```bash
marie flow --uses flow.yml
```

## Complete example: Document processor

Here's a more realistic example that simulates document processing:

### executor.py

```python
from marie import Executor, requests
from docarray import DocList, BaseDoc
from typing import Optional
import time

class DocumentInput(BaseDoc):
    """Input document with text content."""
    content: str = ''
    doc_type: str = 'unknown'

class DocumentOutput(BaseDoc):
    """Output document with extracted data."""
    content: str = ''
    doc_type: str = 'unknown'
    word_count: int = 0
    processing_time_ms: float = 0.0
    extracted_data: Optional[dict] = None

class DocumentProcessor(Executor):
    """Processes documents and extracts basic information."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processed_count = 0

    @requests(on='/extract')
    def extract(
        self,
        docs: DocList[DocumentInput],
        parameters: dict = None,
        **kwargs
    ) -> DocList[DocumentOutput]:
        results = DocList[DocumentOutput]()

        for doc in docs:
            start_time = time.time()

            # Simulate processing
            words = doc.content.split()
            word_count = len(words)

            # Extract basic data
            extracted = {
                'first_word': words[0] if words else None,
                'last_word': words[-1] if words else None,
                'contains_numbers': any(c.isdigit() for c in doc.content),
            }

            processing_time = (time.time() - start_time) * 1000
            self.processed_count += 1

            results.append(DocumentOutput(
                content=doc.content,
                doc_type=doc.doc_type,
                word_count=word_count,
                processing_time_ms=processing_time,
                extracted_data=extracted,
            ))

        return results

    @requests(on='/stats')
    def stats(self, **kwargs):
        return {
            'processed_count': self.processed_count,
            'status': 'healthy'
        }
```

### flow.yml

```yaml
jtype: Flow
version: '1'
with:
  port: 54321
  protocol: [grpc, http]
  port_expose: [54321, 54322]
executors:
  - name: document-processor
    uses: DocumentProcessor
    py_modules:
      - executor.py
```

### client.py

```python
from marie import Client
from docarray import DocList, BaseDoc
from typing import Optional

class DocumentInput(BaseDoc):
    content: str = ''
    doc_type: str = 'unknown'

class DocumentOutput(BaseDoc):
    content: str = ''
    doc_type: str = 'unknown'
    word_count: int = 0
    processing_time_ms: float = 0.0
    extracted_data: Optional[dict] = None

# Connect to the service
client = Client(host='localhost', port=54321, protocol='http')

# Process some documents
docs = DocList[DocumentInput]([
    DocumentInput(
        content='Invoice #12345 from Acme Corp for $1,500.00',
        doc_type='invoice'
    ),
    DocumentInput(
        content='Dear Customer, Thank you for your purchase.',
        doc_type='letter'
    ),
])

# Send to extract endpoint
results = client.post(
    on='/extract',
    inputs=docs,
    return_type=DocList[DocumentOutput]
)

# Display results
for result in results:
    print(f'\n--- Document ({result.doc_type}) ---')
    print(f'Content: {result.content}')
    print(f'Word count: {result.word_count}')
    print(f'Processing time: {result.processing_time_ms:.2f}ms')
    print(f'Extracted data: {result.extracted_data}')
```

Run the flow and client:

```bash
# Terminal 1: Start the service
marie flow --uses flow.yml

# Terminal 2: Run the client
python client.py
```

Expected output:

```text
--- Document (invoice) ---
Content: Invoice #12345 from Acme Corp for $1,500.00
Word count: 7
Processing time: 0.05ms
Extracted data: {'first_word': 'Invoice', 'last_word': '$1,500.00', 'contains_numbers': True}

--- Document (letter) ---
Content: Dear Customer, Thank you for your purchase.
Word count: 6
Processing time: 0.03ms
Extracted data: {'first_word': 'Dear', 'last_word': 'purchase.', 'contains_numbers': False}
```

## Using HTTP directly

You can also send requests using curl or any HTTP client:

```bash
curl -X POST http://localhost:54322/extract \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"content": "Test document content", "doc_type": "test"}
    ]
  }'
```

## Project structure

A typical Marie-AI project looks like this:

```text
my-project/
├── executors/
│   ├── __init__.py
│   ├── processor.py
│   └── extractor.py
├── flows/
│   ├── dev.yml
│   └── prod.yml
├── tests/
│   └── test_executor.py
├── requirements.txt
└── README.md
```

## Next steps

Now that you have a working pipeline:

1. **Learn about Executors**: Build custom processing logic → [Executor guide](../guides/executor.md)
2. **Orchestrate with Flows**: Create complex pipelines → [Flow guide](../guides/flow.md)
3. **Understand the architecture**: See how components fit together → [Architecture overview](../guides/architecture-overview.md)
4. **Deploy to production**: Run in Docker or Kubernetes → [Deployment guides](./deployment/index.md)

## Troubleshooting

### Port already in use

If you see "Address already in use", change the port:

```python
f = Flow(port=54322)  # Use a different port
```

### Module not found

Ensure your executor file is in the Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Connection refused

Make sure the Flow is running before sending requests. Check the terminal for the "Flow is ready" message.
