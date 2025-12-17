---
sidebar_position: 1
---

# Architecture overview

Marie-AI is a document processing framework built on a microservice architecture. This guide explains the core concepts and how they work together to process documents.

## Core concepts

Marie-AI is built around four primary components:

| Component | Description |
| --------- | ----------- |
| **Executor** | A microservice that processes documents |
| **Flow** | Orchestrates multiple Executors into a pipeline |
| **Gateway** | Entry point that routes requests to Executors |
| **Client** | Sends requests to the Gateway |

```text
┌─────────┐     ┌─────────┐     ┌──────────────────────────────────┐
│         │     │         │     │              Flow                │
│ Client  │────▶│ Gateway │────▶│  ┌──────────┐   ┌──────────┐    │
│         │     │         │     │  │Executor 1│──▶│Executor 2│    │
└─────────┘     └─────────┘     │  └──────────┘   └──────────┘    │
                                └──────────────────────────────────┘
```

### Executor

An [Executor](./executor.md) is the fundamental building block. Each Executor:

- Runs as an independent microservice
- Processes documents through decorated methods (`@requests`)
- Can be scaled independently (replicas, shards)
- Communicates via gRPC, HTTP, or WebSocket

Marie-AI provides specialized Executor base classes:

| Class | Purpose |
| ----- | ------- |
| `Executor` | Base class for general processing |
| `MarieExecutor` | Adds GPU monitoring, storage, and health checks |
| `DocumentAnnotatorExecutor` | Specialized for document extraction |

### Flow

A [Flow](./flow.md) connects Executors into a processing pipeline:

- Defines the order of document processing
- Manages Executor lifecycle (start, stop, scale)
- Provides a unified API through the Gateway
- Supports complex topologies (parallel, branching)

### Gateway

The Gateway is the entry point for all requests:

- Routes requests to the appropriate Executors
- Supports multiple protocols (gRPC, HTTP, WebSocket)
- Handles load balancing across Executor replicas
- Exposes health check endpoints

MarieGateway extends the base Gateway with service discovery, job scheduling, real-time events (SSE), and capacity management. See the [Gateway guide](./gateway.md) for details.

### Client

The Client sends requests to the Gateway:

```python
from marie import Client

c = Client(host='localhost', port=54321)
result = c.post(on='/extract', inputs=document)
```

## Document processing pipeline

Marie-AI is optimized for document processing workflows. A typical pipeline processes documents through multiple stages:

```text
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Ingest    │──▶│     OCR     │──▶│  Classify   │──▶│   Extract   │
│  Document   │   │   Engine    │   │  Document   │   │    Data     │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
       │                                                      │
       ▼                                                      ▼
┌─────────────┐                                       ┌─────────────┐
│   Storage   │◀──────────────────────────────────────│    Store    │
│    Layer    │                                       │   Results   │
└─────────────┘                                       └─────────────┘
```

### Pipeline stages

1. **Document ingestion**: Load documents from S3, local storage, or API
2. **Page bursting**: Split multi-page documents into individual pages
3. **Pre-processing**: Clean images, deskew, enhance quality
4. **OCR**: Extract text and layout information
5. **Classification**: Determine document type and structure
6. **Extraction**: Extract structured data using rules or AI models
7. **Post-processing**: Validate, transform, and store results

## Extract engine

The Extract engine is Marie-AI's specialized system for document understanding.

### Components

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Extract Engine                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Readers   │  │  Annotators │  │   Parsers   │             │
│  │             │  │             │  │             │             │
│  │ • MetaReader│  │ • LLM       │  │ • Region    │             │
│  │ • PDF       │  │ • Regex     │  │ • Markdown  │             │
│  │ • Image     │  │ • Embedding │  │ • JSON      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              UnstructuredDocument                        │   │
│  │  • Lines with spatial indexing (R-tree)                 │   │
│  │  • Regions and tables                                   │   │
│  │  • Metadata and confidence scores                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Readers

Readers load documents and extract raw content:

- **MetaReader**: Coordinates reading from multiple sources
- Supports PDF, images, and multi-page documents

### Annotators

Annotators extract structured information from documents:

| Type | Description |
| ---- | ----------- |
| `LLMAnnotator` | Uses language models for extraction |
| `RegexAnnotator` | Pattern-based extraction |
| `FaissHybridAnnotator` | Embedding-based similarity matching |
| `LLMTableAnnotator` | Specialized for table extraction |

### UnstructuredDocument

The core data structure for extracted content:

```python
from marie.extract.structures import UnstructuredDocument

# UnstructuredDocument contains:
# - Lines with text, bounding boxes, and metadata
# - Spatial index for efficient line lookup
# - Regions (headers, footers, tables)
# - Extraction results
```

## Storage layer

Marie-AI provides integrated storage for persistence and asset tracking.

### PostgreSQL storage

Store extraction results and document metadata:

```yaml
storage:
  psql:
    enabled: true
    hostname: localhost
    port: 5432
    database: marie
    username: marie
    password: ${POSTGRES_PASSWORD}
```

### Asset tracking

Track document lineage and processing history:

```python
from marie.executor.mixin import StorageMixin

class MyExecutor(MarieExecutor, StorageMixin):
    def __init__(self, storage: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.setup_storage(
            storage_enabled=True,
            storage_conf=storage['psql'],
            asset_tracking_enabled=True
        )
```

## Document types

Marie-AI uses DocArray for document handling with specialized types:

| Type | Description |
| ---- | ----------- |
| `AssetKeyDoc` | Reference to a document with asset key and pages |
| `MarieDoc` | Image document with tags for OCR |
| `BatchableMarieDoc` | OCR results with words and bounding boxes |
| `StorageDoc` | Document for persistence |
| `OutputDoc` | Job status and response |

### Document flow

```python
from marie.api.docs import AssetKeyDoc
from docarray import DocList

# Input: Reference to document in storage
input_doc = AssetKeyDoc(
    asset_key='s3://bucket/document.pdf',
    pages=[1, 2, 3]
)

# Process through Flow
result = flow.post('/extract', inputs=DocList[AssetKeyDoc]([input_doc]))
```

## Request/response flow

Understanding how requests flow through the system:

```text
1. Client sends request to Gateway
   ┌─────────┐
   │ Client  │──────────────────────────────────────────────────┐
   └─────────┘                                                   │
                                                                 ▼
2. Gateway routes to first Executor                       ┌─────────┐
   ┌─────────┐                                           │ Gateway │
   │Executor1│◀──────────────────────────────────────────└─────────┘
   └─────────┘
        │
        ▼
3. Documents flow through pipeline
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │Executor1│────▶│Executor2│────▶│Executor3│
   └─────────┘     └─────────┘     └─────────┘
                                        │
                                        ▼
4. Response returns to Client                             ┌─────────┐
   ┌─────────┐                                           │ Gateway │
   │ Client  │◀──────────────────────────────────────────└─────────┘
   └─────────┘
```

### Job tracking

For long-running operations, Marie-AI tracks job status:

```python
parameters = {
    'job_id': 'unique-job-id',
    'ref_id': 'document-reference',
    'ref_type': 'invoice',
    'payload': {
        'op_params': {
            'key': 'invoice_extractor',
            'layout': 'default'
        }
    }
}

result = client.post('/extract', inputs=docs, parameters=parameters)
```

## Deployment architecture

### Single node

```text
┌────────────────────────────────────────────────┐
│                   Single Node                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Gateway │  │Executor1│  │Executor2│        │
│  │  :54321 │  │  :54322 │  │  :54323 │        │
│  └─────────┘  └─────────┘  └─────────┘        │
└────────────────────────────────────────────────┘
```

### Kubernetes cluster

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Gateway   │  │  Executor   │  │  Executor   │             │
│  │   Service   │  │   Pod x3    │  │   Pod x2    │             │
│  │             │  │  (replicas) │  │  (replicas) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│                   ┌─────────────┐                               │
│                   │  PostgreSQL │                               │
│                   │   Storage   │                               │
│                   └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Protocol support

Marie-AI supports multiple communication protocols:

| Protocol | Use case |
| -------- | -------- |
| **gRPC** | High-performance internal communication |
| **HTTP** | REST API for external integrations |
| **WebSocket** | Real-time streaming and updates |

Configure protocols in your Flow:

```yaml
jtype: Flow
with:
  protocol: [grpc, http, websocket]
  port: [54321, 54322, 54323]
```

## Next steps

- Build your first [Executor](./executor.md)
- Orchestrate with [Flows](./flow.md)
- Deploy to [Docker](../getting-started/deployment/docker.md) or [Kubernetes](../getting-started/deployment/kubernetes.md)
